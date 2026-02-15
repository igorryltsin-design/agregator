# Миграция SQLite -> PostgreSQL

Документ описывает безопасный переход Agregator с SQLite на PostgreSQL.

## 1. Когда миграция обязательна

Рекомендуется переходить на PostgreSQL, если:

- увеличился объем данных и запросов;
- требуется более стабильная многопользовательская работа;
- нужен production-режим с резервированием и управляемыми backup-стратегиями.

## 2. Подготовка

1. Остановить запись в систему на время миграции.
2. Создать резервную копию SQLite:

```bash
cp catalogue.db backups/catalogue_before_pg_$(date +%Y%m%d_%H%M%S).db
```

3. Подготовить PostgreSQL и пользователя БД.
4. Проверить подключение:

```bash
psql "postgresql://agregator:agregator@localhost:5432/agregator" -c "select 1;"
```

5. Установить переменную окружения:

```bash
export DATABASE_URL="postgresql://agregator:agregator@localhost:5432/agregator"
```

## 3. Перенос данных

Рекомендуемый инструмент: `pgloader`.

```bash
pgloader "sqlite:///$(pwd)/catalogue.db" "$DATABASE_URL"
```

Если `pgloader` недоступен, используйте ETL-процедуру и тестовый импорт.

## 4. Синхронизация схемы (Alembic)

После переноса обязательно привести схему к актуальному состоянию:

```bash
alembic upgrade head
```

## 5. Проверка результата

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

Проверьте дополнительно:

- авторизацию;
- список документов;
- поиск;
- AI-поиск;
- админ-страницы.

## 6. Переключение приложения на PostgreSQL

- Обновить `DATABASE_URL`/`SQLALCHEMY_DATABASE_URI`.
- Перезапустить приложение/контейнер.
- Подтвердить, что backend работает с `postgresql` dialect.

## 7. Rollback-план

При неуспехе:

1. Вернуть переменные подключения к SQLite.
2. Восстановить `catalogue.db` из backup.
3. Перезапустить приложение.
4. Проверить ключевые пользовательские сценарии.

## 8. Рекомендации после миграции

- Настроить регулярные PostgreSQL backup (`pg_dump`).
- Добавить мониторинг времени запросов и размера БД.
- Оставить SQLite backup до подтверждения стабильности прод-режима.
