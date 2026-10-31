
#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

cd "$(dirname "${BASH_SOURCE[0]}")"

MODE="${1:-online}"
[[ "$MODE" == "online" || "$MODE" == "--offline" ]] || {
  echo "Использование: bash setup-mac.sh [--offline]"
  exit 2
}

mkdir -p logs storage/uploads storage/results
LOG="logs/setup-mac-$(date +%Y%m%d-%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

trap 'status=$?; if [ "$status" -ne 0 ]; then
  echo "Ошибка запуска. Журнал: $LOG" >&2
fi' EXIT

info() {
  printf '\n==> %s\n' "$1"
}

fail() {
  echo "ОШИБКА: $1" >&2
  exit 1
}

confirm() {
  local answer
  printf "%s [y/N]: " "$1"
  read -r answer
  [[ "$answer" == "y" || "$answer" == "Y" ]] ||
    fail "Операция отменена"
}

set_env() {
  local key="$1"
  local value="$2"

  if grep -q "^${key}=" .env; then
    sed -i '' "s/^${key}=.*/${key}=${value}/" .env
  else
    printf '%s=%s\n' "$key" "$value" >> .env
  fi
}

info "Проверка системы"

[[ "$(uname -s)" == "Darwin" ]] ||
  fail "Требуется macOS"

[[ "$(uname -m)" == "arm64" ]] ||
  fail "Требуется нативный терминал Apple Silicon (arm64)"

[[ -f compose.yaml && -f pyproject.toml ]] ||
  fail "Запустите сценарий из корня проекта"

[[ -f scripts/prepare_online.sh ]] ||
  fail "Отсутствует scripts/prepare_online.sh"

[[ -f scripts/start_offline.sh ]] ||
  fail "Отсутствует scripts/start_offline.sh"

command -v docker >/dev/null ||
  fail "Docker не установлен"

docker info >/dev/null 2>&1 ||
  fail "Запустите Docker Desktop"

docker compose version

info "Настройка окружения"

if [[ ! -f .env ]]; then
  [[ -f .env.example ]] ||
    fail "Отсутствует .env.example"
  cp .env.example .env
fi

chmod 600 .env

if ! grep -q '^POSTGRES_PASSWORD=.' .env || grep -q '^POSTGRES_PASSWORD=REPLACE_ME_WITH_A_LOCAL_PASSWORD$' .env; then
  command -v openssl >/dev/null ||
    fail "Не найден openssl"

  PASSWORD="$(openssl rand -hex 24)"
  set_env "POSTGRES_PASSWORD" "$PASSWORD"
  unset PASSWORD
fi

grep -q '^POSTGRES_PASSWORD=[^[:space:]]' .env ||
  fail "Укажите POSTGRES_PASSWORD в .env"

set_env "APP_UID" "$(id -u)"
set_env "APP_GID" "$(id -g)"

docker compose config --quiet

if [[ "$MODE" == "online" ]]; then
  info "Проверка инструментов разработки"

  export PATH="/opt/homebrew/bin:$HOME/.local/bin:$PATH"

  command -v brew >/dev/null ||
    fail "Установите Homebrew с https://brew.sh"

  PACKAGES=()

  command -v pyenv >/dev/null ||
    PACKAGES+=(pyenv)

  command -v pipx >/dev/null ||
    PACKAGES+=(pipx)

  command -v sha256sum >/dev/null ||
    PACKAGES+=(coreutils)

  if [[ "${#PACKAGES[@]}" -gt 0 ]]; then
    echo "Требуется установка: ${PACKAGES[*]}"
    confirm "Разрешить установку через Homebrew?"
    brew install "${PACKAGES[@]}"
  fi

  if brew list --versions coreutils >/dev/null 2>&1; then
    export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
  fi

  export PATH="$HOME/.local/bin:$PATH"

  if ! pyenv prefix 3.12.8 >/dev/null 2>&1; then
    confirm "Установить Python 3.12.8?"
    pyenv install 3.12.8
  fi

  if ! command -v poetry >/dev/null; then
    confirm "Установить Poetry 2.2.1?"
    pipx install poetry==2.2.1
  fi

  command -v poetry >/dev/null ||
  fail "Poetry не найден"

  poetry --version

  poetry env use "$(pyenv prefix 3.12.8)/bin/python"

  poetry run python --version

  info "Подготовка зависимостей и Docker-образов"

  bash scripts/prepare_online.sh

  for FILE in \
    poetry.lock \
    offline/docker-images.tar \
    offline/ocr-models.tar.gz \
    offline/SHA256SUMS
  do
    [[ -s "$FILE" ]] ||
      fail "Не создан обязательный файл: $FILE"
  done

  info "Запуск контейнеров"

  docker compose up -d --no-build --pull never
else
  info "Автономный запуск"

  command -v sha256sum >/dev/null ||
    fail "Для автономной проверки нужен sha256sum"

  for FILE in \
    poetry.lock \
    offline/docker-images.tar \
    offline/ocr-models.tar.gz \
    offline/SHA256SUMS
  do
    [[ -s "$FILE" ]] ||
      fail "Отсутствует файл автономного комплекта: $FILE"
  done

  bash scripts/start_offline.sh
fi

info "Проверка сервисов"

docker compose ps

PORT="$(sed -n 's/^API_PORT=//p' .env | tail -n 1)"
PORT="${PORT:-8000}"

READY=0

for i in $(seq 1 30); do
  if curl -fsS --max-time 5 \
    "http://127.0.0.1:${PORT}/health" \
    >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  docker compose logs --tail=100 api worker db
  fail "API не ответил на проверку состояния"
fi

info "Проверка обработки изображения"

if [[ "$PORT" == "8000" ]]; then
  bash scripts/smoke_http.sh
else
  echo "HTTP-тест пропущен: нестандартный API_PORT=$PORT"
fi

docker compose logs --no-color --tail=150 \
  api worker db > logs/docker-last.log

info "Запуск завершён"

echo "API: http://127.0.0.1:${PORT}"
echo "Результаты: storage/results/"
echo "Журнал: $LOG"
echo "Логи Docker: logs/docker-last.log"
