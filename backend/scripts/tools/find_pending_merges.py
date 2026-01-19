"""
Скрипт для поиска кандидатов на объединение кластеров и сохранения их в JSON файл.

Использование:
    python backend/scripts/tools/find_pending_merges.py --person-id 4 --max-source-size 4 --max-distance 0.3

Или через run_face.ps1 (рекомендуется):
    .\backend\scripts\run_face.ps1 backend/scripts/tools/find_pending_merges.py --person-id 4
"""
import sys
import json
import argparse
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.logic.face_recognition import find_optimal_clusters_to_merge_in_person


def main():
    parser = argparse.ArgumentParser(description="Найти кандидатов на объединение кластеров")
    parser.add_argument("--person-id", type=int, required=True, help="ID персоны")
    parser.add_argument("--max-source-size", type=int, default=4, help="Максимальный размер маленького кластера (по умолчанию: 4)")
    parser.add_argument("--max-distance", type=float, default=0.3, help="Максимальное расстояние между кластерами (по умолчанию: 0.3)")
    parser.add_argument("--output-dir", type=str, default=None, help="Директория для сохранения JSON (по умолчанию: backend/data)")
    
    args = parser.parse_args()
    
    # Определяем директорию для сохранения JSON
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / "backend" / "data"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"pending_merges_person_{args.person_id}.json"
    
    print(f"Поиск кандидатов на объединение для персоны {args.person_id}...")
    print(f"  max_source_size: {args.max_source_size}")
    print(f"  max_distance: {args.max_distance}")
    
    try:
        suggestions = find_optimal_clusters_to_merge_in_person(
            person_id=args.person_id,
            max_source_size=args.max_source_size,
            max_distance=args.max_distance,
        )
        
        print(f"\nНайдено предложений: {len(suggestions)}")
        
        # Сохраняем в JSON
        result = {
            "person_id": args.person_id,
            "max_source_size": args.max_source_size,
            "max_distance": args.max_distance,
            "suggestions": suggestions,
            "total": len(suggestions),
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        print(f"\nРезультаты сохранены в: {output_file}")
        
        if suggestions:
            print(f"\nПервые 10 предложений:")
            for i, s in enumerate(suggestions[:10], 1):
                print(f"  {i}. Кластер #{s['source_cluster_id']} ({s['source_cluster_size']} фото) -> "
                      f"Кластер #{s['target_cluster_id']} ({s['target_cluster_size']} фото), "
                      f"расстояние: {s['distance']:.4f}")
        
        # Код возврата 0 при успехе
        return 0
        
    except Exception as e:
        print(f"\nОшибка: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
