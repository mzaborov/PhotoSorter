#!/usr/bin/env python3
"""
Удаление записей старых прогонов из files_manual_labels и file_groups.

Удаляем записи для pipeline_run_id: 18, 20, 21 (старые прогоны с NULL file_id).
"""

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

from backend.common.db import get_connection


def cleanup_old_runs(conn, old_run_ids: list[int], dry_run: bool = False):
    """Удалить записи для старых прогонов."""
    cur = conn.cursor()
    
    # Статистика перед удалением
    print(f"\nСтатистика перед удалением:")
    
    for run_id in old_run_ids:
        # files_manual_labels
        cur.execute("SELECT COUNT(*) as cnt FROM files_manual_labels WHERE pipeline_run_id = ?", (run_id,))
        files_manual_count = cur.fetchone()["cnt"]
        
        # file_groups
        cur.execute("SELECT COUNT(*) as cnt FROM file_groups WHERE pipeline_run_id = ?", (run_id,))
        file_groups_count = cur.fetchone()["cnt"]
        
        # file_group_persons
        cur.execute("SELECT COUNT(*) as cnt FROM file_group_persons WHERE pipeline_run_id = ?", (run_id,))
        file_group_persons_count = cur.fetchone()["cnt"]
        
        print(f"  pipeline_run_id={run_id}:")
        print(f"    files_manual_labels: {files_manual_count} записей")
        print(f"    file_groups: {file_groups_count} записей")
        print(f"    file_group_persons: {file_group_persons_count} записей")
    
    if dry_run:
        print(f"\n[DRY RUN] Записи не будут удалены")
        return
    
    # Удаляем записи
    total_deleted = 0
    
    for run_id in old_run_ids:
        print(f"\nУдаление записей для pipeline_run_id={run_id}...")
        
        # files_manual_labels
        cur.execute("DELETE FROM files_manual_labels WHERE pipeline_run_id = ?", (run_id,))
        deleted = cur.rowcount
        total_deleted += deleted
        print(f"  files_manual_labels: удалено {deleted} записей")
        
        # file_groups
        cur.execute("DELETE FROM file_groups WHERE pipeline_run_id = ?", (run_id,))
        deleted = cur.rowcount
        total_deleted += deleted
        print(f"  file_groups: удалено {deleted} записей")
        
        # file_group_persons
        cur.execute("DELETE FROM file_group_persons WHERE pipeline_run_id = ?", (run_id,))
        deleted = cur.rowcount
        total_deleted += deleted
        print(f"  file_group_persons: удалено {deleted} записей")
    
    print(f"\n✅ Всего удалено записей: {total_deleted}")
    return total_deleted


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Удаление записей старых прогонов")
    parser.add_argument("--dry-run", action="store_true", help="Только показать, что будет сделано")
    parser.add_argument("--yes", action="store_true", help="Автоматически подтвердить удаление (без интерактивного запроса)")
    parser.add_argument("--run-ids", type=str, help="Список pipeline_run_id через запятую (по умолчанию: 18,20,21)")
    args = parser.parse_args()
    
    # Определяем старые прогоны
    if args.run_ids:
        old_run_ids = [int(x.strip()) for x in args.run_ids.split(",")]
    else:
        old_run_ids = [18, 20, 21]  # Старые прогоны с NULL file_id
    
    print("=" * 70)
    print("УДАЛЕНИЕ ЗАПИСЕЙ СТАРЫХ ПРОГОНОВ")
    print("=" * 70)
    print(f"Старые прогоны: {old_run_ids}")
    print(f"Режим: {'DRY RUN' if args.dry_run else 'ВЫПОЛНЕНИЕ'}")
    
    if not args.dry_run:
        if not args.yes:
            response = input(f"\n⚠️  ВНИМАНИЕ: Будут удалены все записи для прогонов {old_run_ids}!\nПродолжить? (yes/no): ")
            if response.lower() != "yes":
                print("Отменено.")
                return 1
        else:
            print(f"\n⚠️  ВНИМАНИЕ: Будут удалены все записи для прогонов {old_run_ids}!")
            print("Продолжаем (--yes указан)...")
    
    conn = get_connection()
    try:
        cleanup_old_runs(conn, old_run_ids, dry_run=args.dry_run)
        
        if not args.dry_run:
            conn.commit()
            print(f"\n✅ Удаление завершено успешно!")
        else:
            print(f"\n[DRY RUN] Удаление не выполнено. Используйте без --dry-run для выполнения.")
        
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Ошибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
