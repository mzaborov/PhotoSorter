/**
 * Единый модуль карточки фотографий для отображения и редактирования фото с rectangles
 * Используется на всех страницах проекта (faces.html, person_detail.html, face_cluster_detail.html)
 */

(function() {
  'use strict';

  // Глобальное состояние карточки
  let currentState = {
    file_id: null,
    file_path: null,
    list_context: null,
    highlight_rectangle: null,
    rectangles: [],
    mode: 'archive', // 'archive' | 'sorting'
    pipeline_run_id: null,
    showRectangles: true
  };

  /**
   * Открывает карточку фотографий
   * @param {Object} options - Параметры открытия
   * @param {number|null} options.file_id - ID файла из таблицы files (приоритет)
   * @param {string|null} options.file_path - Путь к файлу (fallback)
   * @param {Object|null} options.list_context - Контекст списка для навигации
   * @param {Object|null} options.highlight_rectangle - Rectangle для выделения
   */
  function openPhotoCard(options) {
    if (!options) {
      console.error('[photo_card] openPhotoCard: options is required');
      return;
    }

    // Инициализация состояния
    currentState = {
      file_id: options.file_id || null,
      file_path: options.file_path || null,
      list_context: options.list_context || null,
      highlight_rectangle: options.highlight_rectangle || null,
      rectangles: [],
      mode: 'archive',
      pipeline_run_id: null,
      showRectangles: true
    };

    // Если передан list_context, берем file_id и file_path из него
    if (currentState.list_context && currentState.list_context.items) {
      const currentIndex = currentState.list_context.current_index || 0;
      const currentItem = currentState.list_context.items[currentIndex];
      if (currentItem) {
        currentState.file_id = currentItem.file_id || currentState.file_id;
        currentState.file_path = currentItem.file_path || currentState.file_path;
      }
    }

    if (!currentState.file_id && !currentState.file_path) {
      console.error('[photo_card] openPhotoCard: file_id or file_path is required');
      return;
    }

    // Определяем pipeline_run_id из глобальной переменной или URL
    currentState.pipeline_run_id = window.pipelineRunId || getPipelineRunIdFromUrl();

    // Определяем режим (archive/sorting)
    if (currentState.file_path) {
      currentState.mode = currentState.file_path.startsWith('disk:/Фото') ? 'archive' : 'sorting';
    } else {
      // Если только file_id, нужно запросить путь из БД (будет сделано при загрузке)
      currentState.mode = 'sorting'; // По умолчанию
    }

    // Показываем модальное окно
    const modal = document.getElementById('photoCardModal');
    if (!modal) {
      console.error('[photo_card] Modal element not found. Make sure photo_card.html is included.');
      return;
    }

    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');

    // Загружаем данные
    loadPhotoCardData();
  }

  /**
   * Закрывает карточку фотографий
   */
  function closePhotoCard() {
    const modal = document.getElementById('photoCardModal');
    if (modal) {
      modal.style.display = 'none';
      modal.setAttribute('aria-hidden', 'true');
    }

    // Очищаем состояние
    currentState = {
      file_id: null,
      file_path: null,
      list_context: null,
      highlight_rectangle: null,
      rectangles: [],
      mode: 'archive',
      pipeline_run_id: null,
      showRectangles: true
    };
  }

  /**
   * Загружает данные для карточки (изображение и rectangles)
   */
  async function loadPhotoCardData() {
    try {
      // Обновляем путь в заголовке
      const pathElement = document.getElementById('photoCardPath');
      if (pathElement) {
        pathElement.textContent = currentState.file_path || `file_id: ${currentState.file_id}`;
      }

      // Загружаем изображение
      await loadImage();

      // Загружаем rectangles (только для sorting режима)
      if (currentState.mode === 'sorting' && currentState.pipeline_run_id) {
        await loadRectangles();
      }
    } catch (error) {
      console.error('[photo_card] Error loading data:', error);
    }
  }

  /**
   * Загружает изображение
   */
  async function loadImage() {
    const imgElement = document.getElementById('photoCardImg');
    const videoElement = document.getElementById('photoCardVideo');
    
    if (!imgElement || !videoElement) {
      console.error('[photo_card] Image/video elements not found');
      return;
    }

    if (!currentState.file_path) {
      console.error('[photo_card] file_path is required for loading image');
      return;
    }

    // Генерируем URL для изображения
    let imageUrl = '';
    if (currentState.file_path.startsWith('disk:')) {
      // YaDisk
      const encodedPath = encodeURIComponent(currentState.file_path);
      imageUrl = `/api/yadisk/preview-image?size=XL&path=${encodedPath}`;
    } else if (currentState.file_path.startsWith('local:')) {
      // Локальный файл
      const encodedPath = encodeURIComponent(currentState.file_path);
      imageUrl = `/api/local/preview?path=${encodedPath}`;
    } else {
      console.error('[photo_card] Unknown file path format:', currentState.file_path);
      return;
    }

    // Определяем, это изображение или видео
    const isVideo = currentState.file_path.match(/\.(mp4|avi|mov|mkv|webm)$/i);
    
    if (isVideo) {
      imgElement.style.display = 'none';
      videoElement.style.display = 'block';
      videoElement.src = imageUrl;
    } else {
      imgElement.style.display = 'block';
      videoElement.style.display = 'none';
      imgElement.src = imageUrl;
      
      // Ждем загрузки изображения перед отрисовкой rectangles
      imgElement.onload = function() {
        if (currentState.showRectangles) {
          drawRectangles();
        }
      };
    }
  }

  /**
   * Загружает rectangles через API
   */
  async function loadRectangles() {
    if (!currentState.pipeline_run_id) {
      console.warn('[photo_card] pipeline_run_id is required for loading rectangles');
      return;
    }

    try {
      const params = new URLSearchParams();
      params.append('pipeline_run_id', currentState.pipeline_run_id);
      if (currentState.file_id) {
        params.append('file_id', currentState.file_id);
      } else if (currentState.file_path) {
        params.append('path', currentState.file_path);
      }

      const response = await fetch(`/api/faces/rectangles?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      if (data.ok && data.rectangles) {
        currentState.rectangles = data.rectangles;
        drawRectangles();
        updateRectanglesList();
      }
    } catch (error) {
      console.error('[photo_card] Error loading rectangles:', error);
    }
  }

  /**
   * Отрисовывает rectangles на изображении
   */
  function drawRectangles() {
    if (!currentState.showRectangles) {
      return;
    }

    const imgElement = document.getElementById('photoCardImg');
    const canvas = document.getElementById('photoCardCanvas');
    const imgWrap = document.getElementById('photoCardImgWrap');
    
    if (!imgElement || !canvas || !imgWrap) {
      return;
    }

    // Очищаем старые rectangles
    const oldRects = imgWrap.querySelectorAll('.photo-card-rectangle');
    oldRects.forEach(el => el.remove());

    if (!imgElement.complete || imgElement.naturalWidth === 0) {
      return;
    }

    // Получаем размеры изображения
    const imgRect = imgElement.getBoundingClientRect();
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;

    if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
      return;
    }

    // Масштабируем координаты
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;

    // Рисуем каждый rectangle
    currentState.rectangles.forEach((rect, index) => {
      const bbox = rect.bbox || {};
      const x = (bbox.x || 0) * scaleX;
      const y = (bbox.y || 0) * scaleY;
      const w = (bbox.w || 0) * scaleX;
      const h = (bbox.h || 0) * scaleY;

      // Определяем цвет rectangle
      let color = 'rgba(250, 204, 21, 0.3)'; // Желтый по умолчанию (кластеры)
      let borderColor = 'rgba(250, 204, 21, 1)';
      
      // Проверяем, это выделенный rectangle?
      if (currentState.highlight_rectangle) {
        const isHighlighted = 
          (currentState.highlight_rectangle.type === 'face_rectangle' && rect.id === currentState.highlight_rectangle.id) ||
          (currentState.highlight_rectangle.type === 'person_rectangle' && rect.person_rectangle_id === currentState.highlight_rectangle.id);
        
        if (isHighlighted) {
          color = 'rgba(11, 87, 208, 0.3)'; // Синий для выделенного
          borderColor = 'rgba(11, 87, 208, 1)';
        }
      }

      // Создаем элемент rectangle
      const rectElement = document.createElement('div');
      rectElement.className = 'photo-card-rectangle';
      rectElement.style.position = 'absolute';
      rectElement.style.left = `${x}px`;
      rectElement.style.top = `${y}px`;
      rectElement.style.width = `${w}px`;
      rectElement.style.height = `${h}px`;
      rectElement.style.border = `2px solid ${borderColor}`;
      rectElement.style.backgroundColor = color;
      rectElement.style.pointerEvents = 'auto';
      rectElement.style.cursor = 'pointer';
      rectElement.style.zIndex = '1000';
      rectElement.dataset.rectIndex = index;

      imgWrap.appendChild(rectElement);
    });
  }

  /**
   * Обновляет список rectangles внизу карточки
   */
  function updateRectanglesList() {
    const rectList = document.getElementById('photoCardRectList');
    if (!rectList) {
      return;
    }

    rectList.innerHTML = '';

    currentState.rectangles.forEach((rect, index) => {
      const pill = document.createElement('div');
      pill.className = 'rectpill';
      pill.textContent = rect.person_name || `Rectangle ${index + 1}`;
      pill.style.cursor = 'pointer';
      pill.dataset.rectIndex = index;

      // Клик по плашке - выделение rectangle
      pill.addEventListener('click', function() {
        highlightRectangle(index);
      });

      rectList.appendChild(pill);
    });
  }

  /**
   * Выделяет rectangle по индексу
   */
  function highlightRectangle(index) {
    const rect = currentState.rectangles[index];
    if (!rect) {
      return;
    }

    // Обновляем highlight_rectangle
    currentState.highlight_rectangle = {
      type: 'face_rectangle',
      id: rect.id
    };

    // Перерисовываем rectangles
    drawRectangles();
  }

  /**
   * Получает pipeline_run_id из URL
   */
  function getPipelineRunIdFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('pipeline_run_id') ? parseInt(params.get('pipeline_run_id')) : null;
  }

  // Инициализация обработчиков событий
  function initEventHandlers() {
    // Закрытие по клику на кнопку
    const closeBtn = document.getElementById('photoCardClose');
    if (closeBtn) {
      closeBtn.addEventListener('click', closePhotoCard);
    }

    // Закрытие по клику вне карточки
    const modal = document.getElementById('photoCardModal');
    if (modal) {
      modal.addEventListener('click', function(e) {
        if (e.target === modal) {
          closePhotoCard();
        }
      });
    }

    // Закрытие по Escape
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        const modal = document.getElementById('photoCardModal');
        if (modal && modal.style.display !== 'none') {
          closePhotoCard();
        }
      }
    });

    // Переключение видимости rectangles
    const toggleBtn = document.getElementById('photoCardToggleRectangles');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', function() {
        currentState.showRectangles = !currentState.showRectangles;
        toggleBtn.textContent = currentState.showRectangles ? 'Скрыть квадраты' : 'Показать квадраты';
        if (currentState.showRectangles) {
          drawRectangles();
        } else {
          const oldRects = document.querySelectorAll('.photo-card-rectangle');
          oldRects.forEach(el => el.remove());
        }
      });
    }
  }

  // Инициализация при загрузке DOM
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initEventHandlers);
  } else {
    initEventHandlers();
  }

  // Экспортируем функцию openPhotoCard в глобальную область
  window.openPhotoCard = openPhotoCard;
  window.closePhotoCard = closePhotoCard;

})();
