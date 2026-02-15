# Руководство по настройке Agregator

Документ описывает полный цикл подготовки окружения: от проверки требований до первичной настройки приложений и интеграции сервисов LLM/OCR. Используйте его как базу для обучения администраторов и инженеров сопровождения.

## 1. Минимальные требования
- 4 CPU и 8 ГБ RAM (рекомендуется 6+ CPU и 12 ГБ RAM, если активно используете OCR, Whisper и RAG).
- 15–30 ГБ на системном диске для приложения, библиотеки, кешей и резервных копий.
- Docker 24+ и Docker Compose v2 (для сценариев контейнеризации).
- Python 3.11 (при установке без Docker), Node.js 20 (для сборки фронтендов), npm 10.
- Доступ к модели LLM (LM Studio или совместимый API), если используются AI-функции.

## 2. Подготовка окружения
1. **Склонируйте репозиторий** или распакуйте подготовленный архив:
   ```bash
   git clone https://github.com/<org>/Agregator.git
   cd Agregator
   ```
2. **Проверьте наличие системных зависимостей**. Для Debian/Ubuntu:
   ```bash
   sudo apt-get update
   sudo apt-get install -y ffmpeg poppler-utils tesseract-ocr tesseract-ocr-rus
   ```
   В контейнере эти пакеты устанавливаются автоматически из `docker/system-packages.txt`.
3. **Подготовьте директории данных** (на хосте):
   ```bash
   mkdir -p library backups logs models
   ```
4. **Создайте или обновите `.env`**. В Docker-режиме значения хранятся в `docker/app.env`. Для индивидуальных окружений используйте копию `docker/app.dev.env`.

## 3. Сценарий A — запуск в Docker
1. **Выберите целевую платформу**. Для развёртывания на amd64 используйте:
   ```bash
   docker buildx build --platform linux/amd64 -t agregator:amd64 . --load
   ```
2. **Сконфигурируйте переменные окружения**. Основные параметры:
   - `FLASK_SECRET_KEY` — уникальный ключ приложения.
   - `DEFAULT_ADMIN_USER`, `DEFAULT_ADMIN_PASSWORD` — будут сброшены при первом запуске.
   - `LMSTUDIO_API_BASE`, `LMSTUDIO_MODEL`, `LMSTUDIO_API_KEY` — адрес и ключ LLM.
   - `TRANSCRIBE_ENABLED`, `TRANSCRIBE_MODEL_PATH` — включение Whisper (`faster-whisper`).
   Дополнительные переменные смотрите в `docker/app.env`.
3. **Запустите Compose-стек**:
   ```bash
   docker compose up -d
   ```
   По умолчанию откроется порт `5050`. Проверьте логи:
   ```bash
   docker compose logs -f agregator
   ```
   Постоянные директории автоматически примонтированы:
   - `./library:/data` — библиотека.
   - `./backups:/app/backups` — резервные копии и экспорт.
   - `./models:/app/models` — кэш моделей (faster-whisper, LLM).
4. **Обновление**:
   ```bash
   docker compose pull
   docker compose up -d --force-recreate agregator
   ```
5. **Резервная копия контейнерного образа**:
   ```bash
   docker save -o agregator-amd64.tar agregator:amd64
   ```

## 4. Сценарий B — запуск без Docker
1. Создайте виртуальное окружение:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Установите зависимости:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   npm --prefix frontend install && npm --prefix frontend run build
   npm --prefix AiWord install && npm --prefix AiWord run build
   ```
3. Настройте переменные окружения (`cp docker/app.dev.env .env` и отредактируйте значения).
4. Запустите приложение:
   ```bash
   flask --app app run --debug
   ```
   или
   ```bash
   python app.py
   ```
5. Для планировщиков и фона используйте systemd/supervisord. Готовый unit смотрите в `deploy/ansible/roles/agregator/templates/agregator.service.j2`.

## 5. Первичная настройка после запуска
1. **Создайте администратора** (если учетная запись по умолчанию не подходит):
   - Выполните POST-запрос: `POST /api/admin/users` (через Swagger или `scripts/create_admin.py`).
   - Смените пароль первоначальной записи в UI.
2. **Добавьте LLM и ключи**:
   - Интерфейс: `/admin/settings/ai`.
   - Проверьте связь: кнопка «Тест соединения».
3. **Подготовьте модели faster-whisper**:
   - При первом обращении к транскрибации Agregator автоматически скачает модель по алиасу (`small`, `base` и т.д.) в каталог `models/faster-whisper/…`.
   - Если окружение работает офлайн, заранее выполните:
     ```bash
     python -m huggingface_hub download Systran/faster-whisper-small \
       --local-dir models/faster-whisper/Systran__faster-whisper-small \
       --local-dir-use-symlinks False
     ```
   - Проверьте, что в каталоге лежат файлы `model.bin*` размером десятки мегабайт (плейсхолдер git-lfs на 134 байта означает, что модель нужно скачать заново).
4. **Настройте OCR/Tesseract**:
   - Включите нужные языки (`OCR_LANGS`).
   - Проверьте `scripts/test_ocr.sh path/to/sample.pdf`.
5. **Настройте резервные копии**:
   - Точка монтирования `/app/backups`.
   - Используйте `scripts/export_aiword.sh` для подготовки автономного пакета.
6. **Подключите LM Studio / OpenAI**:
   - Проверьте доступность `http://host.docker.internal:1234/v1/models`.
   - Укажите модель (`LMSTUDIO_MODEL`) и ключ (`LMSTUDIO_API_KEY`).

### 5.7. Настройка OSINT-поиска через внешний API
Если стандартный поисковый модуль работает на локальном браузере и часто блокируется капчей, можно переключиться на сторонний API.
- `OSINT_SEARCH_API_URL` — адрес сервиса, который принимает JSON с полями `query`, `engine`, `locale`, `region`, `safe`, `max_results` и возвращает массив `items` или `results` (элемент — объект с `title`, `url`, `snippet` и/или `score`).
- `OSINT_SEARCH_API_METHOD` — HTTP-метод (`GET` или `POST`, по умолчанию `POST`) для запросов к API.
- `OSINT_SEARCH_API_KEY` — (необязательно) ключ, передаётся в заголовке `Authorization: Bearer …`.
После этого Agregator будет посылать запросы к указанному API и напрямую использовать возвращённые ссылки, что помогает обойти Google/Yandex и снизить число ручных капч. При необходимости оставьте `OSINT_RETRY_USER_AGENTS` и `OSINT_RETRY_PROXIES`, чтобы менять user-agent или прокси после подтверждения «Я не робот».

## 6. Проверка после установки
- Откройте `http://<host>:5050/app` и убедитесь, что UI доступен.
- Загрузите тестовый документ, проверьте распознавание текста, карточку и работу поисковой выдачи.
- Выполните быструю диагностику:
  ```bash
  docker compose exec agregator python scripts/diagnostics.py
  ```
  (Скрипт выводит статус базы, списки настроенных коллекций и состояние очередей.)

## 7. Типичные проблемы и решения
- **Контейнер не стартует** — проверьте `FLASK_SECRET_KEY`, наличие миграций (`SearchService.ensure_support` запускается автоматически).
- **Нет доступа к LLM** — убедитесь, что с хоста контейнера виден сервис (`host.docker.internal` для macOS/Windows, `172.17.0.1` для Linux).
- **Whisper выдаёт `Unsupported model binary version`** — в каталоге `models/faster-whisper/...` остались плейсхолдеры git-lfs. Скачайте модель через `python -m huggingface_hub download ...` (см. раздел 5.3) и убедитесь, что файлы `.bin` весят десятки мегабайт.
- **Whisper не создаёт расшифровку** — проверьте `TRANSCRIBE_ENABLED=true` и доступность модели (`small`, `medium`, путь к кастомной модели).
- **PDF без текста** — увеличьте `PDF_OCR_PAGES` или включите `SUMMARIZE_AUDIO`, `KEYWORDS_TO_TAGS_ENABLED`.

## 8. Дополнительные материалы
- `docs/admin_handbook.md` — типовые рабочие процессы и чек-листы.
- `docs/training_resources.md` — план обучающих сессий и ссылки на записи.
- `docs/deployment.md` — развертывание в Docker/Ansible.
- `docs/architecture.md` — обзор модулей и взаимодействий.
- `docs/postgres_migration.md` — пошаговая миграция SQLite -> PostgreSQL и команды Alembic.
- `scripts/migrate_sqlite_to_postgres.sh` — helper-скрипт миграции (dry-run/run).

Обновляйте документ после каждого релиза: новые переменные окружения, изменения схемы БД, поведение фронтенда/фоновых задач.
