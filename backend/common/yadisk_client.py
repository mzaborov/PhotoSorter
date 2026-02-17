import os
from pathlib import Path

import yadisk
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent  # backend/common
BACKEND_DIR = BASE_DIR.parent              # backend
REPO_ROOT = BACKEND_DIR.parent             # корень проекта


def _load_env() -> None:
    """
    Загружает переменные окружения из файла с секретами.
    Проверяем: корень проекта (secrets.env), backend (secrets.env), backend/common (secrets.env/.env).
    """
    for base in (REPO_ROOT, BACKEND_DIR, BASE_DIR):
        secrets_env = base / "secrets.env"
        if secrets_env.exists():
            load_dotenv(secrets_env)
            return
    for base in (REPO_ROOT, BACKEND_DIR, BASE_DIR):
        dot_env = base / ".env"
        if dot_env.exists():
            load_dotenv(dot_env)
            return


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
