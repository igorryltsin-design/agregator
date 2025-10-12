# Локальная разработка и деплой

Этот документ описывает практические сценарии запуска Agregator с помощью Docker Compose на целевой платформе **linux/amd64** и без него.

## Предварительные требования
- Docker 24+ и Docker Compose v2.
- 6–8 ГБ RAM на хосте (OCR/Whisper при первом старте могут потребовать больше времени).
- 15+ ГБ свободного места для библиотеки, кешей и резервных копий.
- Опционально: LM Studio или другой OpenAI-совместимый сервер, доступный с хоста.

## Файлы окружения
- `docker/app.env` — базовые настройки контейнера (production-like).
- `docker/app.dev.env` — dev-профиль: включает debug, отключает тяжёлые фоновые задачи.
- `docker/app.test.env` — профиль для юнит-тестов.

> ⚠️ Не храните в репозитории секреты. Если нужно переопределить значения, используйте `docker-compose.override.yml` или экспортируйте переменные в среду перед запуском (`export LMSTUDIO_API_KEY=...`).

## Сценарии Docker Compose
### Production-like (docker-compose.yml)
```bash
docker compose build
# первый запуск
docker compose up -d
# обновление
docker compose pull
docker compose up -d --force-recreate agregator
```

### Локальная разработка с горячей перезагрузкой
`docker-compose.dev.yml` наследует базовый сервис и добавляет монтирование исходников + команду `flask run --reload`.

```bash
# 1. Соберите образ (однократно):
docker compose -f docker-compose.dev.yml build

# 2. Поднимите контейнер с дев-настройками:
docker compose -f docker-compose.dev.yml up --remove-orphans

# 3. Откройте http://localhost:5050/app. Изменения Python-кода применяются автоматически.
```

> Чтобы переопределить dev-настройки, создайте `docker-compose.override.yml` и добавьте блок `services.agregator-dev.environment` со своими значениями.

### Запуск юнит-тестов в контейнере
`docker-compose.test.yml` наслаивается на базовый сервис и запускает `pytest`.

```bash
# Выполнить тесты и удалить контейнер после завершения
docker compose -f docker-compose.test.yml run --rm agregator-test
```

### Обновление фронтенда
Образ автоматически собирает `frontend` и `AiWord`. Для локальной разработки без Docker можно:
```bash
npm --prefix frontend install
npm --prefix frontend run build
npm --prefix AiWord install
npm --prefix AiWord run build
```
Статические файлы появятся в `frontend/dist` и `AiWord/dist`.

### Резервные копии и каталоги данных
- `./library` → `/data` — директория с материалами (настраивается через `SCAN_ROOT`).
- `./backups` → `/app/backups` — архивы БД/метаданных.
- `./logs` → `/app/logs` — журналы `agregator.log` и фоновых задач.

Регулярно копируйте `library` и `backups` на внешний носитель.

## Альтернативный запуск без Docker
Инструкции в разделе «Установка и запуск» README остаются актуальны: используйте virtualenv, `pip install -r requirements.txt`, `python app.py`. Для Watch-mode можно установить `pip install watchdog` и запускать `flask --app app --debug run`.

## Типовые проблемы
- **Контейнер падает сразу после старта.** Проверьте логи `docker compose logs agregator` и корректность `FLASK_SECRET_KEY`.
- **LM Studio недоступен из контейнера.** На macOS/Windows используйте `http://host.docker.internal:1234/v1`.
- **OCR не работает в образе.** Убедитесь, что `docker/system-packages.txt` содержит нужные языковые пакеты (`tesseract-ocr-rus` и т.п.), пересоберите образ.

Дополнительно смотрите `docs/architecture.md` и README для обзора функций и API.
