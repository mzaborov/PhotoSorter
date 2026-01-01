import os
from pathlib import Path

import yadisk
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    """
    Загружает переменные окружения из файла с секретами.
    Поддерживаем `secrets.env` (основной) и `.env` (fallback), ничего не печатаем.
    """
    secrets_env = BASE_DIR / "secrets.env"
    dot_env = BASE_DIR / ".env"

    if secrets_env.exists():
        load_dotenv(secrets_env)
    elif dot_env.exists():
        load_dotenv(dot_env)


def get_disk() -> yadisk.YaDisk:
    _load_env()
    token = os.getenv("YADISK_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("YADISK_ACCESS_TOKEN не найден в secrets.env/.env")

    disk = yadisk.YaDisk(token=token)
    if not disk.check_token():
        raise RuntimeError("Недействительный YADISK_ACCESS_TOKEN")

    return disk


if __name__ == "__main__":
    # Ручная проверка, что токен валиден (без вывода токена)
    d = get_disk()
    print("OK: token valid =", d.check_token())
