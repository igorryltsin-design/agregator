# Миграция SQLite -> PostgreSQL

Документ для перехода со старого формата `catalogue.db` (SQLite) на PostgreSQL с сохранением совместимости.

## 1) Подготовка

- Создайте резервную копию SQLite:
  - `cp catalogue.db backups/catalogue_before_pg_$(date +%Y%m%d_%H%M%S).db`
- Убедитесь, что PostgreSQL доступен:
  - `psql "postgresql://agregator:agregator@localhost:5432/agregator" -c "select 1;"`
- Укажите URL PostgreSQL:
  - `export DATABASE_URL="postgresql://agregator:agregator@localhost:5432/agregator"`

## 2) Перенос данных из SQLite

Рекомендуемый инструмент: `pgloader`.

- macOS:
  - `brew install pgloader`
- Debian/Ubuntu:
  - `sudo apt-get install -y pgloader`

Запуск:

```bash
pgloader "sqlite:///$(pwd)/catalogue.db" "$DATABASE_URL"
```

Если `pgloader` недоступен, оставьте SQLite как legacy-режим и выполните миграцию отдельно через ETL.

## 3) Применение Alembic

После переноса данных обязательно выровняйте схему:

```bash
alembic upgrade head
```

Проверка:

```bash
python - <<'PY'
from app import app, db
from models import File, Tag, Collection
with app.app_context():
    print("dialect:", db.engine.dialect.name)
    print("files:", db.session.query(File).count())
    print("tags:", db.session.query(Tag).count())
    print("collections:", db.session.query(Collection).count())
PY
```

## 4) Совместимость со старым форматом

- В UI оставлен выбор типа БД:
  - `SQLite (legacy .db)` — старый формат.
  - `PostgreSQL` — основной режим.
- Импорт `.db` и прямой backup `.db` продолжают работать для SQLite.
- Для PostgreSQL в UI поддержаны:
  - backup в `.dump` или `.sql`,
  - restore из `.dump` или `.sql`.

## 5) Rollback

- Для быстрого возврата:
  - верните `DATABASE_URL`/`SQLALCHEMY_DATABASE_URI` на SQLite,
  - восстановите файл `catalogue.db` из `backups/`,
  - перезапустите приложение.
