-- Проверка персон для конкретного файла
-- Замените путь на нужный

-- 1. Прямые привязки через face_labels
SELECT 
    'face_labels (прямая)' AS source,
    p.id AS person_id,
    p.name AS person_name,
    fr.id AS face_rectangle_id,
    fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h
FROM face_labels fl
JOIN persons p ON p.id = fl.person_id
JOIN face_rectangles fr ON fr.id = fl.face_rectangle_id
WHERE fr.file_path = 'local:C:\tmp\Photo\_faces\IMG-20240311-WA0001.jpeg'
  AND fr.run_id = (SELECT face_run_id FROM pipeline_runs WHERE id = 26)
ORDER BY p.name, fr.id;

-- 2. Привязки через кластеры
SELECT DISTINCT
    'через кластеры' AS source,
    p.id AS person_id,
    p.name AS person_name,
    fr_file.id AS face_rectangle_id,
    fr_file.bbox_x, fr_file.bbox_y, fr_file.bbox_w, fr_file.bbox_h,
    fc.id AS cluster_id,
    fc.name AS cluster_name
FROM persons p
JOIN face_labels fl_cluster ON fl_cluster.person_id = p.id
-- Находим кластеры, где есть лица с face_labels для этой персоны
JOIN face_cluster_members fcm_labeled ON fcm_labeled.face_rectangle_id = fl_cluster.face_rectangle_id
JOIN face_clusters fc ON fc.id = fcm_labeled.cluster_id
-- Находим ВСЕ лица в этих кластерах (включая лица из нашего файла)
JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = fc.id
JOIN face_rectangles fr_file ON fr_file.id = fcm_all.face_rectangle_id
WHERE fr_file.file_path = 'local:C:\tmp\Photo\_faces\IMG-20240311-WA0001.jpeg'
  AND fr_file.run_id = (SELECT face_run_id FROM pipeline_runs WHERE id = 26)
  AND COALESCE(fr_file.ignore_flag, 0) = 0
  -- Исключаем прямые привязки
  AND NOT EXISTS (
      SELECT 1 FROM face_labels fl_direct
      WHERE fl_direct.face_rectangle_id = fr_file.id
        AND fl_direct.person_id = fl_cluster.person_id
  )
ORDER BY p.name, fr_file.id, fc.id;

-- 3. Все лица в файле
SELECT 
    fr.id AS face_rectangle_id,
    fr.bbox_x, fr.bbox_y, fr.bbox_w, fr.bbox_h,
    fr.face_num,
    COALESCE(fr.ignore_flag, 0) AS ignore_flag
FROM face_rectangles fr
WHERE fr.file_path = 'local:C:\tmp\Photo\_faces\IMG-20240311-WA0001.jpeg'
  AND fr.run_id = (SELECT face_run_id FROM pipeline_runs WHERE id = 26)
ORDER BY fr.face_num;

-- 4. Кластеры, в которых находятся лица из файла
SELECT DISTINCT
    fc.id AS cluster_id,
    fc.name AS cluster_name,
    fc.run_id AS cluster_run_id,
    fc.archive_scope,
    fr.id AS face_rectangle_id,
    fr.face_num
FROM face_rectangles fr
JOIN face_cluster_members fcm ON fcm.face_rectangle_id = fr.id
JOIN face_clusters fc ON fc.id = fcm.cluster_id
WHERE fr.file_path = 'local:C:\tmp\Photo\_faces\IMG-20240311-WA0001.jpeg'
  AND fr.run_id = (SELECT face_run_id FROM pipeline_runs WHERE id = 26)
ORDER BY fc.id, fr.face_num;

-- 5. Все персоны в этих кластерах (через face_labels)
SELECT DISTINCT
    fc.id AS cluster_id,
    fc.name AS cluster_name,
    p.id AS person_id,
    p.name AS person_name,
    COUNT(DISTINCT fl.face_rectangle_id) AS labeled_faces_in_cluster
FROM face_rectangles fr_file
JOIN face_cluster_members fcm_file ON fcm_file.face_rectangle_id = fr_file.id
JOIN face_clusters fc ON fc.id = fcm_file.cluster_id
JOIN face_cluster_members fcm_all ON fcm_all.cluster_id = fc.id
JOIN face_labels fl ON fl.face_rectangle_id = fcm_all.face_rectangle_id
JOIN persons p ON p.id = fl.person_id
WHERE fr_file.file_path = 'local:C:\tmp\Photo\_faces\IMG-20240311-WA0001.jpeg'
  AND fr_file.run_id = (SELECT face_run_id FROM pipeline_runs WHERE id = 26)
GROUP BY fc.id, fc.name, p.id, p.name
ORDER BY fc.id, p.name;
