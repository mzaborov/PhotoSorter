"""
Тестовый скрипт для проверки функции find_optimal_clusters_to_merge_in_person
"""
import sys
from pathlib import Path

# Добавляем путь к проекту
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from backend.logic.face_recognition import find_optimal_clusters_to_merge_in_person

if __name__ == "__main__":
    person_id = 4
    max_source_size = 4
    max_distance = 0.3
    
    print(f"Вызываем find_optimal_clusters_to_merge_in_person(person_id={person_id}, max_source_size={max_source_size}, max_distance={max_distance})")
    print("=" * 80)
    
    suggestions = find_optimal_clusters_to_merge_in_person(
        person_id=person_id,
        max_source_size=max_source_size,
        max_distance=max_distance,
    )
    
    print(f"\nНайдено предложений: {len(suggestions)}")
    
    if suggestions:
        print(f"\nПервые 10 предложений:")
        for i, s in enumerate(suggestions[:10], 1):
            print(f"  {i}. Кластер #{s['source_cluster_id']} ({s['source_cluster_size']} фото) -> "
                  f"Кластер #{s['target_cluster_id']} ({s['target_cluster_size']} фото), "
                  f"расстояние: {s['distance']:.4f}")
    else:
        print("\nПредложений не найдено!")