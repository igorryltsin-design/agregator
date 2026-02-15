# Развертывание Agregator

Документ описывает рабочие сценарии выката: Docker Compose, запуск без Docker и поставка релизного артефакта.

## 1. Стратегии деплоя

- **Docker Compose (рекомендуется):** быстрый и повторяемый запуск.
- **Bare-metal (Python):** для сред без контейнеров.
- **Релизный пакет + Ansible:** для контролируемого промышленного rollout.

## 2. Docker Compose

### 2.1 Базовые команды

```bash
docker compose build
docker compose up -d
docker compose ps
```

Обновление:

```bash
docker compose pull
docker compose up -d --force-recreate agregator
```

### 2.2 Важные volume-пути

- `./library -> /data` - рабочая библиотека.
- `./backups -> /app/backups` - резервные копии и экспорт.
- `./models -> /app/models` - модели и кэш артефактов.
- `./logs -> /app/logs` - журналы приложения.

### 2.3 Проверка после выката

- Проверить HTTP-доступность `http://<host>:5050/app`.
- Проверить логи `docker compose logs agregator`.
- Выполнить smoke-сценарий: login -> импорт -> поиск -> AI-поиск.

## 3. Локальный сервер без Docker

1. Подготовить virtualenv и зависимости.
2. Собрать frontend (`frontend`, `AiWord`).
3. Запустить `python app.py`.
4. При сервисном режиме использовать systemd/supervisord.

Пример запуска через Flask debug:

```bash
flask --app app run --debug --port 5050
```

## 4. Релизный пакет

В репозитории предусмотрен упаковочный сценарий:

```bash
./scripts/package_release.sh
```

Результат: архив в `dist/` с backend, frontend-сборками и служебными файлами.

## 5. Развертывание через Ansible

Если используется `deploy/ansible`:

1. Подготовить inventory.
2. Передать путь к релизному архиву.
3. Выполнить playbook.

Пример:

```bash
ansible-playbook -i deploy/ansible/inventory deploy/ansible/site.yml \
  -e agregator_release_tarball=/tmp/agregator-<version>.tar.gz \
  -e agregator_release_id=<version>
```

## 6. Регламент обновления

1. Создать резервную копию БД и `runtime_settings.json`.
2. Зафиксировать текущую версию и окружение.
3. Выполнить обновление.
4. Прогнать smoke-тест.
5. Подтвердить работоспособность AI и задач.
6. Обновить внутренний журнал изменений.

## 7. План отката

При неуспешном обновлении:

1. Остановить новую версию.
2. Восстановить предыдущий контейнер/артефакт.
3. Вернуть резервную копию БД.
4. Перезапустить сервис и повторить проверку ключевых функций.

## 8. Эксплуатационные рекомендации

- Выделять отдельную БД PostgreSQL для прод-режима.
- Ограничивать прямой доступ к admin endpoint-ам.
- Настроить ротацию логов и мониторинг диска.
- Регулярно проверять восстановление из резервной копии.
