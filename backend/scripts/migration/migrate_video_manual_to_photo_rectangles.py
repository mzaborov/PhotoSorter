#!/usr/bin/env python3
"""
Миграция: переход с video_manual_frames на photo_rectangles + files.

1. Добавляет frame_idx, frame_t_sec в photo_rectangles
2. Добавляет video_frame1_t_sec, video_frame2_t_sec, video_frame3_t_sec в files
3. Переносит данные из video_manual_frames в photo_rectangles и files
4. Удаляет таблицу video_manual_frames

По плану: универсальная_карточка_видео.plan.md

Использование:
  python migrate_video_manual_to_photo_rectangles.py          # применить
  python migrate_video_manual_to_photo_rectangles.py --dry-run # показать, что будет сделано (без изменений БД)
"""
import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=str(repo_root / "secrets.env"), override=False)
    load_dotenv(dotenv_path=str(repo_root / ".env"), override=False)
except Exception:
    pass

from backend.common.db import get_connection, _ensure_columns, _now_utc_iso


def main() -> int:
    parser = argparse.ArgumentParser(description="Миграция video_manual_frames → photo_rectangles + files")
    parser.add_argument("--dry-run", action="store_true", help="Показать, что будет сделано, без изменений БД")
    args = parser.parse_args()
    dry_run = args.dry_run

    if dry_run:
        print("🔍 Режим --dry-run: изменения БД не применяются\n")
    conn = get_connection()
    cur = conn.cursor()

    # 1. Добавляем колонки в photo_rectangles
    if dry_run:
        print("1. Будет добавлено в photo_rectangles: frame_idx, frame_t_sec")
    else:
        _ensure_columns(conn, "photo_rectangles", {
            "frame_idx": "frame_idx INTEGER",      # 1..3 для кадров видео, NULL для фото
            "frame_t_sec": "frame_t_sec REAL",     # таймкод кадра (сек), NULL для фото
        })
        conn.commit()

    # 2. Добавляем колонки в files
    if dry_run:
        print("2. Будет добавлено в files: video_frame1_t_sec, video_frame2_t_sec, video_frame3_t_sec")
    else:
        _ensure_columns(conn, "files", {
            "video_frame1_t_sec": "video_frame1_t_sec REAL",
            "video_frame2_t_sec": "video_frame2_t_sec REAL",
            "video_frame3_t_sec": "video_frame3_t_sec REAL",
        })
        conn.commit()

    # 3. Миграция данных из video_manual_frames (если таблица существует)
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_manual_frames'")
    if cur.fetchone() is None:
        rows = []
    else:
        cur.execute("SELECT pipeline_run_id, file_id, frame_idx, t_sec, rects_json FROM video_manual_frames ORDER BY pipeline_run_id, file_id, frame_idx")
        rows = cur.fetchall()
    migrated = 0
    for r in rows:
        pipeline_run_id = int(r["pipeline_run_id"])
        file_id = int(r["file_id"])
        frame_idx = int(r["frame_idx"] or 0)
        t_sec = float(r["t_sec"]) if r["t_sec"] is not None else None
        rects_json = r["rects_json"]
        if frame_idx not in (1, 2, 3):
            continue
        if not rects_json or rects_json.strip() in ("", "[]"):
            # Пустые rects — только обновляем files
            if not dry_run:
                cur.execute(
                    f"UPDATE files SET video_frame{frame_idx}_t_sec = ? WHERE id = ?",
                    (t_sec, file_id),
                )
            migrated += 1
            continue

        # Получаем face_run_id
        cur.execute("SELECT face_run_id FROM pipeline_runs WHERE id = ?", (pipeline_run_id,))
        pr_row = cur.fetchone()
        face_run_id = int(pr_row["face_run_id"]) if pr_row and pr_row["face_run_id"] is not None else None
        if face_run_id is None:
            print(f"⚠️ pipeline_run_id={pipeline_run_id} не имеет face_run_id, пропускаем")
            continue

        try:
            obj = json.loads(rects_json)
        except Exception:
            obj = []
        if not isinstance(obj, list):
            obj = []

        now = _now_utc_iso()
        face_index = 0
        for it in obj:
            if not isinstance(it, dict):
                continue
            x = int(it.get("x") or 0)
            y = int(it.get("y") or 0)
            w = int(it.get("w") or 0)
            h = int(it.get("h") or 0)
            if w <= 0 or h <= 0:
                continue
            face_index += 1
            manual_person_id = None
            if it.get("manual_person_id") is not None:
                manual_person_id = int(it.get("manual_person_id"))

            if not dry_run:
                cur.execute(
                    """
                    INSERT INTO photo_rectangles(
                      run_id, file_id, face_index,
                      bbox_x, bbox_y, bbox_w, bbox_h,
                      confidence, presence_score, thumb_jpeg,
                      embedding, manual_person, ignore_flag, created_at,
                      is_manual, manual_created_at, is_face,
                      frame_idx, frame_t_sec, manual_person_id
                    )
                    VALUES(?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0, ?, 1, ?, 1, ?, ?, ?)
                    """,
                    (face_run_id, file_id, face_index, x, y, w, h, now, now, frame_idx, t_sec, manual_person_id),
                )
            migrated += 1

        # Обновляем позицию кадра в files
        if not dry_run:
            cur.execute(
                f"UPDATE files SET video_frame{frame_idx}_t_sec = ? WHERE id = ?",
                (t_sec, file_id),
            )

    if not dry_run:
        conn.commit()
    if dry_run:
        print(f"3. Перенос из video_manual_frames: {migrated} rects/кадров")
    elif migrated > 0:
        print(f"✅ Мигрировано записей: {migrated}")

    # 4. Удаляем таблицу video_manual_frames
    if dry_run:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='video_manual_frames'")
        if cur.fetchone():
            print("4. Будет выполнено: DROP TABLE video_manual_frames")
        else:
            print("4. Таблица video_manual_frames отсутствует, DROP не требуется")
    else:
        cur.execute("DROP TABLE IF EXISTS video_manual_frames")
        conn.commit()
        print("✅ Таблица video_manual_frames удалена")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
