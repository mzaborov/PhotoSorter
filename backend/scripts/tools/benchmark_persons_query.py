#!/usr/bin/env python3
"""
Бенчмарк запроса api_faces_persons_with_files для проверки производительности.
"""

import sys
import time
from pathlib import Path

# Добавляем корень репозитория в путь
repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from backend.common.db import FaceStore, PipelineStore

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Бенчмарк запроса персон")
    parser.add_argument("--pipeline-run-id", type=int, default=26, help="ID прогона pipeline")
    parser.add_argument("--query", type=int, choices=[1, 2, 3, 4], help="Номер запроса для тестирования (1-4), если не указан - все")
    args = parser.parse_args()
    
    ps = PipelineStore()
    try:
        pr = ps.get_run_by_id(run_id=int(args.pipeline_run_id))
    finally:
        ps.close()
    if not pr:
        print(f"❌ Прогон {args.pipeline_run_id} не найден!")
        return
    
    face_run_id = pr.get("face_run_id")
    if not face_run_id:
        print(f"❌ У прогона {args.pipeline_run_id} нет face_run_id!")
        return
    
    face_run_id_i = int(face_run_id)
    root_path = str(pr.get("root_path") or "")
    
    # Формируем root_like
    root_like = None
    if root_path.startswith("disk:"):
        rp = root_path.rstrip("/")
        root_like = rp + "/%"
    else:
        try:
            rp_abs = Path(root_path).resolve()
            rp_abs = str(rp_abs).rstrip("\\/") + "\\"
            root_like = "local:" + rp_abs + "%"
        except Exception:
            root_like = None
    
    fs = FaceStore()
    try:
        conn = fs.conn
        cur = conn.cursor()
        
        # Запрос 1: Через лица (face_labels)
        if not args.query or args.query == 1:
            print("\n" + "="*80)
            print("Запрос 1: Через лица (face_labels)")
            print("="*80)
            where_parts = ["fr.run_id = ?"]
            params = [face_run_id_i]
            if root_like:
                where_parts.append("fr.file_path LIKE ?")
                params.append(root_like)
            where_sql = " AND ".join(where_parts)
            
            query_sql = f"""
                SELECT DISTINCT fl.person_id, p.name AS person_name, COUNT(DISTINCT fr.file_path) AS files_count
                FROM face_labels fl
                JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
                LEFT JOIN persons p ON p.id = fl.person_id
                WHERE {where_sql} AND fl.person_id IS NOT NULL
                GROUP BY fl.person_id, p.name
            """
            
            start_time = time.time()
            cur.execute(query_sql, params)
            results = cur.fetchall()
            elapsed = time.time() - start_time
            
            print(f"Время выполнения: {elapsed:.3f}с")
            print(f"Найдено персон: {len(results)}")
            if len(results) > 0:
                print(f"Примеры: {[(r['person_id'], r['person_name'], r['files_count']) for r in results[:5]]}")
        
        # Запрос 2: Через кластеры (старый вариант)
        if not args.query or args.query == 2:
            print("\n" + "="*80)
            print("Запрос 2: Через кластеры (старый вариант)")
            print("="*80)
            where_parts_cluster = ["fr_cluster.run_id = ?"]
            params_cluster = [face_run_id_i]
            if root_like:
                where_parts_cluster.append("fr_cluster.file_path LIKE ?")
                params_cluster.append(root_like)
            where_sql_cluster = " AND ".join(where_parts_cluster)
            
            query_sql_old = f"""
                SELECT DISTINCT fl_cluster.person_id, p.name AS person_name, COUNT(DISTINCT fr_cluster.file_path) AS files_count
                FROM face_labels fl_cluster
                JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
                JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
                JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = fc.id
                JOIN face_rectangles fr_cluster ON fr_cluster.id = fcm_all.face_rectangle_id
                LEFT JOIN persons p ON p.id = fl_cluster.person_id
                WHERE {where_sql_cluster} 
                  AND fl_cluster.person_id IS NOT NULL
                  AND COALESCE(fr_cluster.ignore_flag, 0) = 0
                  AND (fc.run_id = ? OR fc.archive_scope = 'archive')
                GROUP BY fl_cluster.person_id, p.name
            """
            
            start_time = time.time()
            cur.execute(query_sql_old, params_cluster + [face_run_id_i])
            results_old = cur.fetchall()
            elapsed_old = time.time() - start_time
            
            print(f"Время выполнения (старый): {elapsed_old:.3f}с")
            print(f"Найдено персон: {len(results_old)}")
            
            # Запрос 2: Через кластеры (новый оптимизированный вариант)
            print("\n" + "="*80)
            print("Запрос 2: Через кластеры (новый оптимизированный)")
            print("="*80)
            
            query_sql_new = f"""
                SELECT 
                    person_clusters.person_id,
                    p.name AS person_name,
                    COUNT(DISTINCT fr_cluster.file_path) AS files_count
                FROM (
                    SELECT DISTINCT
                        fl_cluster.person_id,
                        fcm_labeled.cluster_id
                    FROM face_labels fl_cluster
                    JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
                    JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
                    WHERE fl_cluster.person_id IS NOT NULL
                      AND (fc.run_id = ? OR fc.archive_scope = 'archive')
                ) person_clusters
                JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = person_clusters.cluster_id
                JOIN face_rectangles fr_cluster ON fr_cluster.id = fcm_all.face_rectangle_id
                LEFT JOIN persons p ON p.id = person_clusters.person_id
                WHERE {where_sql_cluster}
                  AND COALESCE(fr_cluster.ignore_flag, 0) = 0
                  AND NOT EXISTS (
                      SELECT 1 FROM face_labels fl_direct
                      WHERE fl_direct.face_rectangle_id = fr_cluster.id
                        AND fl_direct.person_id = person_clusters.person_id
                  )
                GROUP BY person_clusters.person_id, p.name
            """
            
            start_time = time.time()
            cur.execute(query_sql_new, [face_run_id_i] + params_cluster)
            results_new = cur.fetchall()
            elapsed_new = time.time() - start_time
            
            print(f"Время выполнения (новый): {elapsed_new:.3f}с")
            print(f"Найдено персон: {len(results_new)}")
            print(f"\nУлучшение: {elapsed_old/elapsed_new:.2f}x быстрее" if elapsed_new > 0 else "Новое время: 0")
        
        # Запрос 3: Через person_rectangles
        if not args.query or args.query == 3:
            print("\n" + "="*80)
            print("Запрос 3: Через person_rectangles")
            print("="*80)
            where_parts2 = ["pr.pipeline_run_id = ?"]
            params2 = [int(args.pipeline_run_id)]
            if root_like:
                where_parts2.append("pr.file_path LIKE ?")
                params2.append(root_like)
            where_sql2 = " AND ".join(where_parts2)
            
            query_sql = f"""
                SELECT DISTINCT pr.person_id, p.name AS person_name, COUNT(DISTINCT pr.file_path) AS files_count
                FROM person_rectangles pr
                LEFT JOIN persons p ON p.id = pr.person_id
                WHERE {where_sql2} AND pr.person_id IS NOT NULL
                GROUP BY pr.person_id, p.name
            """
            
            start_time = time.time()
            cur.execute(query_sql, params2)
            results = cur.fetchall()
            elapsed = time.time() - start_time
            
            print(f"Время выполнения: {elapsed:.3f}с")
            print(f"Найдено персон: {len(results)}")
        
        # Запрос 4: Через file_persons
        if not args.query or args.query == 4:
            print("\n" + "="*80)
            print("Запрос 4: Через file_persons")
            print("="*80)
            where_parts3 = ["fp.pipeline_run_id = ?"]
            params3 = [int(args.pipeline_run_id)]
            if root_like:
                where_parts3.append("fp.file_path LIKE ?")
                params3.append(root_like)
            where_sql3 = " AND ".join(where_parts3)
            
            query_sql = f"""
                SELECT DISTINCT fp.person_id, p.name AS person_name, COUNT(DISTINCT fp.file_path) AS files_count
                FROM file_persons fp
                LEFT JOIN persons p ON p.id = fp.person_id
                WHERE {where_sql3} AND fp.person_id IS NOT NULL
                GROUP BY fp.person_id, p.name
            """
            
            start_time = time.time()
            cur.execute(query_sql, params3)
            results = cur.fetchall()
            elapsed = time.time() - start_time
            
            print(f"Время выполнения: {elapsed:.3f}с")
            print(f"Найдено персон: {len(results)}")
        
    finally:
        fs.close()

if __name__ == "__main__":
    main()
