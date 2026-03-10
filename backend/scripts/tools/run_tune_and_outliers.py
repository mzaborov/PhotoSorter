#!/usr/bin/env python3
"""
Единый скрипт пайплайна: поиск аутлайеров → применение ignore_flag → повторный тюнинг.
Показывает статус каждого шага. Вывод по умолчанию дублируется в лог для анализа.

  python backend/scripts/tools/run_tune_and_outliers.py
  python backend/scripts/tools/run_tune_and_outliers.py --log-file path/to/log.log
"""

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOP_N = 15
LOG_DIR = ROOT / "backend" / "scripts" / "debug" / "_logs"
OUTLIERS_SCRIPT = ROOT / "backend" / "scripts" / "tools" / "find_cluster_outliers.py"
COPY_SNAPSHOT_SCRIPT = ROOT / "backend" / "scripts" / "tools" / "copy_archive_ground_truth_to_experiments.py"
TUNE_SCRIPT = ROOT / "backend" / "scripts" / "tools" / "tune_face_clustering.py"


def _tee(msg: str, log_file, end: str = "\n") -> None:
    print(msg, end=end, flush=True)
    if log_file:
        log_file.write(msg + end)
        log_file.flush()


def run(
    cmd: list[str],
    step_num: int,
    step_total: int,
    title: str,
    log_file,
) -> int:
    _tee("", log_file)
    _tee("=" * 60, log_file)
    _tee(f"  Шаг {step_num}/{step_total}: {title}", log_file)
    _tee("=" * 60, log_file)
    proc = subprocess.Popen(
        [sys.executable] + cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if proc.stdout:
        for line in proc.stdout:
            _tee(line.rstrip("\n\r"), log_file)
    ret = proc.wait()
    if ret != 0:
        _tee(f"Ошибка: выход с кодом {ret}", log_file)
        return ret
    _tee(f"  Шаг {step_num}/{step_total} готов.", log_file)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Пайплайн: аутлайеры → apply → тюнинг (один скрипт со статусом)")
    p.add_argument("--no-dry-run", action="store_true", help="Пропустить шаг 1 (dry-run), сразу apply и тюнинг")
    p.add_argument("--top-n", type=int, default=TOP_N, help="Топ-N худших в каждом кластере (по умолчанию %d)" % TOP_N)
    p.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Путь к лог-файлу (по умолчанию backend/scripts/debug/_logs/tune_and_outliers_YYYYMMDD_HHMMSS.log)",
    )
    p.add_argument("--no-log", action="store_true", help="Не писать лог в файл, только в консоль")
    args = p.parse_args()
    top_n = args.top_n

    log_path = None
    log_file = None
    if not args.no_log:
        log_path = args.log_file
        if log_path is None:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            log_path = LOG_DIR / f"tune_and_outliers_{ts}.log"
        try:
            log_file = open(log_path, "w", encoding="utf-8")
            _tee(f"Лог пайплайна: {log_path}", log_file)
        except OSError as e:
            print(f"Не удалось открыть лог-файл {log_path}: {e}", file=sys.stderr, flush=True)
            log_file = None
            log_path = None

    try:
        _tee("Пайплайн тюнинга кластеризации лиц (аутлайеры + ARI)", log_file)
        _tee("Порядок: dry-run аутлайеров → пометка ignore_flag=1 → тюнинг по очищенному ground truth", log_file)
        if log_path:
            _tee(f"Вывод также сохраняется в: {log_path}", log_file)

        step = 1
        total = 4 if not args.no_dry_run else 3

        if not args.no_dry_run:
            if run(
                [str(OUTLIERS_SCRIPT), "--top-n", str(top_n), "--limit", "50"],
                step, total, "Поиск аутлайеров (dry-run, топ-%d на кластер)" % top_n,
                log_file,
            ) != 0:
                return 1
            step += 1

        if run(
            [str(OUTLIERS_SCRIPT), "--top-n", str(top_n), "--save-to-experiments"],
            step, total, "Аутлайеры → experiments.db (основная БД не меняется)",
            log_file,
        ) != 0:
            return 1
        step += 1

        if run(
            [str(COPY_SNAPSHOT_SCRIPT)],
            step, total, "Копирование ground truth в experiments.db",
            log_file,
        ) != 0:
            return 1
        step += 1

        if run(
            [str(TUNE_SCRIPT)],
            step, total, "Тюнинг кластеризации (ARI по очищенному ground truth)",
            log_file,
        ) != 0:
            return 1

        _tee("", log_file)
        _tee("=" * 60, log_file)
        _tee("  Пайплайн завершён. Итог: рекомендации (eps, min_samples) и ARI выше.", log_file)
        _tee("=" * 60, log_file)
        if log_path:
            _tee(f"Полный вывод сохранён в: {log_path}", log_file)
        return 0
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    sys.exit(main())
