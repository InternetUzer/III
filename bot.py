import asyncio
import logging
import os
import sqlite3
import time
from contextlib import closing
from typing import List, Tuple

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, FSInputFile
from aiogram.enums import ChatAction
from aiogram.client.default import DefaultBotProperties
from dotenv import load_dotenv
import ffmpeg

from openai import OpenAI

load_dotenv()

# --- ENV ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "Ты — помощник, разговаривающий на русском языке. Отвечай кратко и по делу.")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "700"))
STT_MODEL = os.getenv("STT_MODEL", "whisper-1")
HISTORY_MAX_MESSAGES = int(os.getenv("HISTORY_MAX_MESSAGES", "12"))
USE_CONTEXT_DEFAULT = os.getenv("USE_CONTEXT", "true").lower() == "true"

if not TELEGRAM_BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN не задан.")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY не задан.")

# --- OpenAI ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Telegram ---
dp = Dispatcher()
bot = Bot(token=TELEGRAM_BOT_TOKEN, default=DefaultBotProperties(parse_mode="HTML"))

# --- DB (SQLite) ---
DB_PATH = "history.db"
os.makedirs("data", exist_ok=True)

def db_connect():
    return sqlite3.connect(DB_PATH)

def db_init():
    with closing(db_connect()) as conn, conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS messages(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,          -- 'user' | 'assistant'
            content TEXT NOT NULL,
            ts INTEGER NOT NULL
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS user_settings(
            user_id INTEGER PRIMARY KEY,
            use_context INTEGER NOT NULL DEFAULT 1
        )
        """)

def set_use_context(user_id: int, enabled: bool):
    with closing(db_connect()) as conn, conn:
        conn.execute("""
            INSERT INTO user_settings (user_id, use_context)
            VALUES (?, ?)
            ON CONFLICT(user_id) DO UPDATE SET use_context=excluded.use_context
        """, (user_id, 1 if enabled else 0))

def get_use_context(user_id: int) -> bool:
    with closing(db_connect()) as conn, conn:
        cur = conn.execute("SELECT use_context FROM user_settings WHERE user_id=?", (user_id,))
        row = cur.fetchone()
        if row is None:
            # default
            return USE_CONTEXT_DEFAULT
        return bool(row[0])

def add_message(user_id: int, role: str, content: str):
    with closing(db_connect()) as conn, conn:
        conn.execute("INSERT INTO messages(user_id, role, content, ts) VALUES(?,?,?,?)",
                     (user_id, role, content, int(time.time())))

def clear_history(user_id: int):
    with closing(db_connect()) as conn, conn:
        conn.execute("DELETE FROM messages WHERE user_id=?", (user_id,))

def get_history(user_id: int, limit: int) -> List[Tuple[str, str]]:
    with closing(db_connect()) as conn, conn:
        cur = conn.execute("""
            SELECT role, content FROM messages
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (user_id, limit))
        rows = cur.fetchall()[::-1]
        return rows

db_init()

# --- Helpers ---
TG_MAX_TEXT = 4000

async def openai_answer(messages: List[Tuple[str, str]]) -> str:
    """
    messages: список пар (role, content) в формате OpenAI ('system'|'user'|'assistant', текст)
    """
    try:
        inputs = [{"role": r, "content": c} for r, c in messages]
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=inputs,
            max_output_tokens=MAX_TOKENS,
        )
        out = getattr(resp, "output_text", None)
        if isinstance(out, str) and out.strip():
            return out.strip()

        parts = []
        for item in getattr(resp, "output", []) or []:
            if hasattr(item, "content") and item.content:
                for c in item.content:
                    if getattr(c, "type", "") == "output_text":
                        parts.append(c.text)
        return ("".join(parts) or "Не удалось получить ответ.").strip()
    except Exception as e:
        logging.exception("OpenAI error")
        return f"Произошла ошибка при обращении к модели: {e}"

async def transcribe_file(path: str) -> str:
    """
    Отправляет аудио-файл в OpenAI для распознавания речи.
    Возвращает распознанный текст.
    """
    try:
        with open(path, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=STT_MODEL,
                file=f,
            )
        # у SDK 1.x возвращается объект с текстом в поле 'text'
        text = getattr(tr, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        return "Не удалось распознать речь."
    except Exception as e:
        logging.exception("Transcription error")
        return f"Ошибка распознавания: {e}"

def convert_ogg_to_mp3(src_path: str, dst_path: str):
    """
    Конвертация .ogg/.oga (opus) из Telegram в .mp3 через ffmpeg.
    """
    (
        ffmpeg
        .input(src_path)
        .output(dst_path, format='mp3', acodec='libmp3lame', ac=1, ar='16000')
        .overwrite_output()
        .run(quiet=True)
    )

def build_message_stack(user_id: int, user_prompt: str) -> List[Tuple[str, str]]:
    """
    Формирует историю для запроса в модель.
    """
    msgs: List[Tuple[str, str]] = [("system", SYSTEM_PROMPT)]
    if get_use_context(user_id):
        hist = get_history(user_id, HISTORY_MAX_MESSAGES)
        msgs.extend(hist)
    msgs.append(("user", user_prompt))
    return msgs

async def reply_long(message: Message, text: str):
    if len(text) <= TG_MAX_TEXT:
        await message.answer(text)
        return
    chunk = []
    total = 0
    for para in text.split("\n"):
        if total + len(para) + 1 > TG_MAX_TEXT:
            await message.answer("\n".join(chunk))
            chunk = [para]
            total = len(para) + 1
        else:
            chunk.append(para)
            total += len(para) + 1
    if chunk:
        await message.answer("\n".join(chunk))

# --- Handlers ---
@dp.message(CommandStart())
async def cmd_start(message: Message):
    text = (
        "Привет! Я бот для общения с моим ИИ‑помощником.\n\n"
        "Отправьте текст или голосовое — я отвечу.\n\n"
        "Команды:\n"
        "/reset — очистить историю диалога\n"
        "/ctx — включить/выключить использование контекста"
    )
    await message.answer(text)

@dp.message(Command("reset"))
async def cmd_reset(message: Message):
    clear_history(message.from_user.id)
    await message.answer("История диалога очищена.")

@dp.message(Command("ctx"))
async def cmd_ctx(message: Message):
    current = get_use_context(message.from_user.id)
    set_use_context(message.from_user.id, not current)
    status = "включено" if not current else "выключено"
    await message.answer(f"Использование контекста: {status}.")

@dp.message(F.text)
async def handle_text(message: Message):
    user_id = message.from_user.id
    user_text = (message.text or "").strip()
    if not user_text:
        return

    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)

    # контекст
    msgs = build_message_stack(user_id, user_text)
    reply = await openai_answer(msgs)

    # сохранить историю
    add_message(user_id, "user", user_text)
    add_message(user_id, "assistant", reply)

    await reply_long(message, reply)

@dp.message(F.voice | F.audio)
async def handle_voice(message: Message):
    """
    Поддержка:
      - voice (OGG/OPUS)
      - audio (может быть m4a, mp3 и т.д.)
    """
    user_id = message.from_user.id
    file_id = None
    if message.voice:
        file_id = message.voice.file_id
    elif message.audio:
        file_id = message.audio.file_id
    if not file_id:
        return

    await bot.send_chat_action(message.chat.id, ChatAction.RECORD_VOICE)

    # скачать файл
    file = await bot.get_file(file_id)
    src_path = f"data/{file.file_unique_id}.ogg"
    dst_path = f"data/{file.file_unique_id}.mp3"
    await bot.download_file(file.file_path, src_path)

    # попытка конвертации в mp3 (иногда whisper принимает и ogg, но стабильнее mp3)
    try:
        convert_ogg_to_mp3(src_path, dst_path)
        audio_path = dst_path
    except Exception:
        audio_path = src_path  # fallback

    # распознаем
    await bot.send_chat_action(message.chat.id, ChatAction.TYPING)
    recognized = await transcribe_file(audio_path)

    # отвечаем как на текст + сохраняем историю
    msgs = build_message_stack(user_id, recognized)
    reply = await openai_answer(msgs)

    add_message(user_id, "user", recognized)
    add_message(user_id, "assistant", reply)

    await reply_long(message, f"🗣️ Распознано: <i>{recognized}</i>\n\n{reply}")

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
