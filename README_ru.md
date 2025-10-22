# Телеграм-бот для общения со мной (OpenAI) — с голосом и контекстом

Функции:
- Текст ↔ ответ модели (OpenAI Responses API).
- **Голосовые/аудио**: распознавание речи (OpenAI `whisper-1`). Просто пришлите голосовое.
- **Хранение контекста** (SQLite): последние N сообщений по пользователю передаются в модель.
- Команды: `/start`, `/reset` (очистить историю), `/ctx` (вкл/выкл использование истории).

## Быстрый старт

1) Python 3.10+
2) Скопируйте `.env.example` в `.env` и заполните токены:
   - `TELEGRAM_BOT_TOKEN`
   - `OPENAI_API_KEY`
   - (опц.) `OPENAI_MODEL` (по умолчанию `gpt-4o`)
3) Установите зависимости и запустите:
   ```bash
   pip install -r requirements.txt
   python bot.py
   ```

### Голосовые заметки

Для конвертации `.ogg` из Telegram в `.mp3` используется `ffmpeg`. Установите его в системе:
- Debian/Ubuntu:
  ```bash
  sudo apt-get update && sudo apt-get install -y ffmpeg
  ```
Docker-образ уже содержит установку.

### Переменные окружения

См. `.env.example`. Ключевые:
- `OPENAI_MODEL` — модель для ответов (например, `gpt-4o`).
- `STT_MODEL` — модель для распознавания (по умолчанию `whisper-1`).
- `HISTORY_MAX_MESSAGES` — сколько последних сообщений передавать в модель (по умолчанию 12).
- `USE_CONTEXT` — `true`/`false` глобально (можно переключать командой `/ctx`).

### Деплой

- **Docker**: `docker build -t tg-bot . && docker run --env-file .env tg-bot`
- Хостинги типа Railway/Render/Fly.io — используйте `Procfile`.

