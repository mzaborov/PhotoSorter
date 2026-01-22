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
    showRectangles: true,
    selectedRectangleIndex: null, // Индекс выделенного rectangle для редактирования
    isDrawing: false, // Режим рисования нового rectangle
    dragState: null, // Состояние drag&drop: {rectIndex, startX, startY, originalX, originalY, originalW, originalH}
    allPersons: [], // Список всех персон для модального окна
    editingRectangleIndex: null // Индекс rectangle, для которого открыто модальное окно выбора персоны
  };

  /**
   * Открывает карточку фотографий
   * @param {Object} options - Параметры открытия
   * @param {number|null} options.file_id - ID файла из таблицы files (приоритет)
   * @param {string|null} options.file_path - Путь к файлу (fallback)
   * @param {number|null} options.pipeline_run_id - ID pipeline run (приоритет)
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
      showRectangles: true,
      selectedRectangleIndex: null,
      isDrawing: false,
      dragState: null,
      allPersons: [],
      editingRectangleIndex: null
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

    // Определяем pipeline_run_id: из options, затем из list_context, затем из глобальной переменной или URL
    currentState.pipeline_run_id = options.pipeline_run_id || null;
    if (!currentState.pipeline_run_id && currentState.list_context && currentState.list_context.api_fallback) {
      currentState.pipeline_run_id = currentState.list_context.api_fallback.params?.pipeline_run_id || null;
    }
    if (!currentState.pipeline_run_id) {
      currentState.pipeline_run_id = window.pipelineRunId || getPipelineRunIdFromUrl();
    }
    
    // Если pipeline_run_id всё ещё нет, но есть run_id в list_context, пытаемся получить через API
    if (!currentState.pipeline_run_id && currentState.list_context && currentState.list_context.items) {
      const currentItem = currentState.list_context.items[currentState.list_context.current_index || 0];
      if (currentItem && currentItem.run_id) {
        // Пытаемся получить pipeline_run_id из API по run_id
        // Но для этого нужен отдельный эндпойнт, пока пропускаем
        console.warn('[photo_card] Has run_id but no pipeline_run_id, rectangles may not load');
      }
    }
    
    console.log('[photo_card] pipeline_run_id:', currentState.pipeline_run_id, 'mode:', currentState.mode);

    // Определяем режим (archive/sorting)
    if (currentState.file_path) {
      currentState.mode = currentState.file_path.startsWith('disk:/Фото') ? 'archive' : 'sorting';
    } else {
      // Если только file_id, нужно запросить путь из БД (будет сделано при загрузке)
      currentState.mode = 'sorting'; // По умолчанию
    }
    
    // Показываем навигацию, если есть list_context
    const navigation = document.getElementById('photoCardNavigation');
    if (navigation && currentState.list_context) {
      navigation.style.display = 'flex';
      updateNavigationPosition();
    } else if (navigation) {
      navigation.style.display = 'none';
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
    
    // Сохраняем контекст для возврата (если есть list_context)
    if (currentState.list_context) {
      // Можно сохранить current_index в sessionStorage для восстановления позиции
      const contextKey = `photoCard_context_${currentState.list_context.source_page}`;
      try {
        sessionStorage.setItem(contextKey, JSON.stringify({
          current_index: currentState.list_context.current_index,
          file_id: currentState.file_id,
          file_path: currentState.file_path
        }));
      } catch (e) {
        console.warn('[photo_card] Failed to save context:', e);
      }
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
      showRectangles: true,
      selectedRectangleIndex: null,
      isDrawing: false,
      dragState: null,
      allPersons: [],
      editingRectangleIndex: null
    };
  }
  
  /**
   * Обновляет позицию в навигации
   */
  function updateNavigationPosition() {
    const positionElement = document.getElementById('photoCardPosition');
    if (!positionElement || !currentState.list_context) return;
    
    const currentIndex = currentState.list_context.current_index || 0;
    const totalCount = currentState.list_context.total_count || 0;
    
    positionElement.textContent = `${currentIndex + 1} из ${totalCount}`;
    
    // Обновляем состояние кнопок навигации
    const prevBtn = document.getElementById('photoCardPrev');
    const nextBtn = document.getElementById('photoCardNext');
    
    if (prevBtn) {
      prevBtn.disabled = currentIndex <= 0;
    }
    if (nextBtn) {
      nextBtn.disabled = currentIndex >= totalCount - 1;
    }
  }
  
  /**
   * Переходит к предыдущему файлу в списке
   */
  async function navigatePrev() {
    if (!currentState.list_context) return;
    
    const currentIndex = currentState.list_context.current_index || 0;
    if (currentIndex <= 0) return;
    
    const newIndex = currentIndex - 1;
    await navigateToIndex(newIndex);
  }
  
  /**
   * Переходит к следующему файлу в списке
   */
  async function navigateNext() {
    if (!currentState.list_context) return;
    
    const currentIndex = currentState.list_context.current_index || 0;
    const totalCount = currentState.list_context.total_count || 0;
    if (currentIndex >= totalCount - 1) return;
    
    const newIndex = currentIndex + 1;
    await navigateToIndex(newIndex);
  }
  
  /**
   * Переходит к файлу по индексу
   */
  async function navigateToIndex(newIndex) {
    if (!currentState.list_context) return;
    
    const items = currentState.list_context.items || [];
    const totalCount = currentState.list_context.total_count || 0;
    
    // Если индекс в пределах загруженных items
    if (newIndex >= 0 && newIndex < items.length) {
      const item = items[newIndex];
      currentState.list_context.current_index = newIndex;
      currentState.file_id = item.file_id || currentState.file_id;
      currentState.file_path = item.file_path || currentState.file_path;
      
      updateNavigationPosition();
      await loadPhotoCardData();
      return;
    }
    
    // Если индекс выходит за границы items, но есть api_fallback
    if (newIndex >= 0 && newIndex < totalCount && currentState.list_context.api_fallback) {
      // Вычисляем нужную страницу
      const pageSize = 60; // Размер страницы по умолчанию
      const page = Math.floor(newIndex / pageSize) + 1;
      const indexInPage = newIndex % pageSize;
      
      try {
        const apiFallback = currentState.list_context.api_fallback;
        const params = new URLSearchParams(apiFallback.params || {});
        params.set('page', page.toString());
        params.set('page_size', pageSize.toString());
        
        const response = await fetch(`${apiFallback.endpoint}?${params.toString()}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        if (data.ok && data.items && data.items[indexInPage]) {
          const item = data.items[indexInPage];
          currentState.list_context.current_index = newIndex;
          currentState.file_id = item.file_id || (item.path ? null : currentState.file_id);
          currentState.file_path = item.path || item.file_path || currentState.file_path;
          
          // Обновляем items в контексте (можно добавить кэширование)
          currentState.list_context.items = data.items;
          
          updateNavigationPosition();
          await loadPhotoCardData();
        }
      } catch (error) {
        console.error('[photo_card] Error navigating to index:', error);
        alert('Ошибка при загрузке файла: ' + error.message);
      }
    }
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

      // Загружаем rectangles (для всех режимов, pipeline_run_id опционален для архивных фото)
      await loadRectangles();
      if (currentState.pipeline_run_id) {
        await checkDuplicates();
      }
      
      // Показываем специальные действия для sorting режима
      const specialActions = document.getElementById('photoCardSpecialActions');
      if (specialActions) {
        specialActions.style.display = currentState.mode === 'sorting' ? 'block' : 'none';
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
        if (currentState.showRectangles && currentState.rectangles.length > 0) {
          drawRectangles();
        }
      };
      
      // Если изображение уже загружено, рисуем сразу
      if (imgElement.complete && imgElement.naturalWidth > 0) {
        if (currentState.showRectangles && currentState.rectangles.length > 0) {
          drawRectangles();
        }
      }
    }
  }

  /**
   * Загружает rectangles через API
   */
  async function loadRectangles() {
    if (!currentState.file_id && !currentState.file_path) {
      console.warn('[photo_card] file_id or file_path is required for loading rectangles');
      return;
    }

    try {
      const params = new URLSearchParams();
      // pipeline_run_id опционален (для архивных фотографий)
      if (currentState.pipeline_run_id) {
        params.append('pipeline_run_id', currentState.pipeline_run_id);
      }
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
      console.log('[photo_card] Loaded rectangles:', data.ok, data.rectangles?.length || 0, 'rectangles');
      if (data.ok && data.rectangles) {
        currentState.rectangles = data.rectangles;
        console.log('[photo_card] Rectangles data sample:', currentState.rectangles.slice(0, 2).map(r => ({
          id: r.id,
          bbox_x: r.bbox_x,
          bbox_y: r.bbox_y,
          bbox_w: r.bbox_w,
          bbox_h: r.bbox_h,
          bbox: r.bbox,
          person_name: r.person_name
        })));
        // Проверяем, загружено ли изображение перед отрисовкой
        const imgElement = document.getElementById('photoCardImg');
        if (imgElement && imgElement.complete && imgElement.naturalWidth > 0) {
          console.log('[photo_card] Image already loaded, drawing rectangles');
          // Небольшая задержка для гарантии, что DOM обновлен
          setTimeout(() => drawRectangles(), 10);
        } else {
          console.log('[photo_card] Image not ready yet, will draw on load');
        }
        updateRectanglesList();
      } else {
        console.warn('[photo_card] No rectangles in response:', data);
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
      console.log('[photo_card] Rectangles hidden by showRectangles flag');
      return;
    }
    
    if (!currentState.rectangles || currentState.rectangles.length === 0) {
      console.log('[photo_card] No rectangles to draw');
      return;
    }

    const imgElement = document.getElementById('photoCardImg');
    const canvas = document.getElementById('photoCardCanvas');
    const imgWrap = document.getElementById('photoCardImgWrap');
    
    if (!imgElement || !canvas || !imgWrap) {
      console.warn('[photo_card] Missing DOM elements for drawing rectangles');
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
      console.warn('[photo_card] Image not loaded yet:', { naturalWidth: imgNaturalWidth, naturalHeight: imgNaturalHeight });
      return;
    }

    // Масштабируем координаты
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;

    console.log('[photo_card] Drawing rectangles:', {
      count: currentState.rectangles.length,
      scale: { x: scaleX, y: scaleY },
      imageSize: { natural: { w: imgNaturalWidth, h: imgNaturalHeight }, display: { w: imgDisplayWidth, h: imgDisplayHeight } }
    });
    
    // Рисуем каждый rectangle
    currentState.rectangles.forEach((rect, index) => {
      // Поддерживаем оба формата: bbox объект или отдельные поля
      let bbox_x, bbox_y, bbox_w, bbox_h;
      if (rect.bbox && typeof rect.bbox === 'object') {
        // Формат: {bbox: {x, y, w, h}}
        bbox_x = rect.bbox.x || 0;
        bbox_y = rect.bbox.y || 0;
        bbox_w = rect.bbox.w || 0;
        bbox_h = rect.bbox.h || 0;
      } else {
        // Формат: {bbox_x, bbox_y, bbox_w, bbox_h}
        bbox_x = rect.bbox_x || 0;
        bbox_y = rect.bbox_y || 0;
        bbox_w = rect.bbox_w || 0;
        bbox_h = rect.bbox_h || 0;
      }
      
      // Пропускаем rectangles с нулевыми размерами
      if (!bbox_w || !bbox_h || bbox_w <= 0 || bbox_h <= 0) {
        console.warn('[photo_card] Skipping rectangle with invalid bbox:', rect.id, {bbox_x, bbox_y, bbox_w, bbox_h});
        return;
      }
      
      // Проверяем, что координаты не слишком большие (возможно, уже в пикселях экрана)
      // Если координаты больше натуральных размеров изображения, возможно они уже в пикселях экрана
      let x, y, w, h;
      if (bbox_x > imgNaturalWidth || bbox_y > imgNaturalHeight) {
        // Координаты уже в пикселях экрана, используем их напрямую
        console.warn('[photo_card] Coordinates seem to be in display pixels, using directly:', { bbox_x, bbox_y, imgNaturalWidth, imgNaturalHeight });
        x = bbox_x;
        y = bbox_y;
        w = bbox_w;
        h = bbox_h;
      } else {
        // Координаты в натуральных размерах, масштабируем
        x = bbox_x * scaleX;
        y = bbox_y * scaleY;
        w = bbox_w * scaleX;
        h = bbox_h * scaleY;
      }
      
      console.log('[photo_card] Drawing rectangle', rect.id, {
        bbox: { x: bbox_x, y: bbox_y, w: bbox_w, h: bbox_h },
        display: { x, y, w, h },
        scale: { x: scaleX, y: scaleY },
        imageSize: { natural: { w: imgNaturalWidth, h: imgNaturalHeight }, display: { w: imgDisplayWidth, h: imgDisplayHeight } }
      });

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

      // Проверяем, это выделенный rectangle?
      const isSelected = currentState.selectedRectangleIndex === index;
      if (isSelected) {
        color = 'rgba(34, 197, 94, 0.3)'; // Зеленый для выделенного
        borderColor = 'rgba(34, 197, 94, 1)';
      }
      
      // Проверяем дубликаты (красный восклицательный знак)
      const isDuplicate = rect.is_duplicate || false;
      
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
      rectElement.style.cursor = currentState.mode === 'sorting' ? 'move' : 'pointer';
      rectElement.style.zIndex = isSelected ? '1001' : '1000';
      rectElement.dataset.rectIndex = index;
      rectElement.dataset.rectId = rect.id || '';
      
      // Добавляем имя персоны на rectangle (если есть)
      if (rect.person_name) {
        const label = document.createElement('div');
        label.className = 'photo-card-rectangle-label';
        label.textContent = (isDuplicate ? '⚠ ' : '') + rect.person_name;
        label.style.position = 'absolute';
        label.style.top = '-20px';
        label.style.left = '0';
        label.style.background = isSelected ? 'rgba(34, 197, 94, 0.95)' : 'rgba(250, 204, 21, 0.95)';
        label.style.color = isSelected ? '#fff' : '#111827';
        label.style.padding = '3px 8px';
        label.style.fontSize = '12px';
        label.style.fontWeight = '700';
        label.style.borderRadius = '4px';
        label.style.whiteSpace = 'nowrap';
        label.style.zIndex = '10002';
        label.style.pointerEvents = 'none';
        if (isDuplicate) {
          label.style.color = '#dc2626'; // Красный для дубликатов
        }
        rectElement.appendChild(label);
      }
      
      // Добавляем rectangle в контейнер изображения
      imgWrap.appendChild(rectElement);
      
      // Добавляем обработчики событий для редактирования (только для sorting режима)
      if (currentState.mode === 'sorting') {
        // Клик - выделение rectangle
        rectElement.addEventListener('click', function(e) {
          e.stopPropagation();
          selectRectangle(index);
        });
        
        // Двойной клик - назначение персоны
        rectElement.addEventListener('dblclick', function(e) {
          e.stopPropagation();
          showPersonDialog(index);
        });
        
        // Drag & drop для перемещения
        rectElement.addEventListener('mousedown', function(e) {
          if (e.button !== 0) return; // Только левая кнопка мыши
          e.preventDefault();
          startDrag(index, e.clientX, e.clientY);
        });
      }

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
      pill.style.display = 'inline-flex';
      pill.style.alignItems = 'center';
      pill.style.gap = '8px';
      pill.style.cursor = 'pointer';
      pill.style.position = 'relative';
      pill.dataset.rectIndex = index;
      
      // Текст с именем персоны или номером
      const text = document.createElement('span');
      text.textContent = (rect.is_duplicate ? '⚠ ' : '') + (rect.person_name || `Rectangle ${index + 1}`);
      if (rect.is_duplicate) {
        text.style.color = '#dc2626';
      }
      pill.appendChild(text);
      
      // Выделение активного rectangle
      if (currentState.selectedRectangleIndex === index) {
        pill.style.background = '#eef2ff';
        pill.style.border = '1px solid #c7d2fe';
      }
      
      // Клик по плашке - выделение rectangle
      pill.addEventListener('click', function(e) {
        if (e.target.closest('.rectpill-action')) return; // Игнорируем клики по кнопкам действий
        selectRectangle(index);
      });
      
      // Двойной клик - назначение персоны (только для sorting режима)
      if (currentState.mode === 'sorting') {
        pill.addEventListener('dblclick', function(e) {
          e.stopPropagation();
          showPersonDialog(index);
        });
        
        // Кнопка действий (выпадающее меню)
        const actionsBtn = document.createElement('button');
        actionsBtn.className = 'rectpill-action';
        actionsBtn.textContent = '⋮';
        actionsBtn.style.background = 'none';
        actionsBtn.style.border = 'none';
        actionsBtn.style.cursor = 'pointer';
        actionsBtn.style.padding = '2px 6px';
        actionsBtn.style.fontSize = '16px';
        actionsBtn.style.lineHeight = '1';
        actionsBtn.addEventListener('click', function(e) {
          e.stopPropagation();
          showRectangleActionsMenu(index, e.target);
        });
        pill.appendChild(actionsBtn);
      }

      rectList.appendChild(pill);
    });
  }

  /**
   * Показывает меню действий для rectangle
   */
  function showRectangleActionsMenu(rectIndex, buttonElement) {
    const rect = currentState.rectangles[rectIndex];
    if (!rect || !rect.id) return;
    
    // Простое меню через confirm/alert (можно улучшить позже)
    const actions = [
      'Назначить персону',
      'Удалить rectangle'
    ];
    
    const choice = prompt('Выберите действие:\n1. Назначить персону\n2. Удалить rectangle\n\nВведите номер:');
    
    if (choice === '1') {
      showPersonDialog(rectIndex);
    } else if (choice === '2') {
      if (confirm('Удалить этот rectangle?')) {
        deleteRectangle(rect.id);
      }
    }
  }


  /**
   * Получает pipeline_run_id из URL
   */
  function getPipelineRunIdFromUrl() {
    const params = new URLSearchParams(window.location.search);
    return params.get('pipeline_run_id') ? parseInt(params.get('pipeline_run_id')) : null;
  }

  /**
   * Выделяет rectangle по индексу
   */
  function selectRectangle(index) {
    currentState.selectedRectangleIndex = index;
    drawRectangles();
    updateRectanglesList();
  }

  /**
   * Начинает drag&drop для перемещения rectangle
   */
  function startDrag(rectIndex, clientX, clientY) {
    const rect = currentState.rectangles[rectIndex];
    if (!rect) return;
    
    const imgElement = document.getElementById('photoCardImg');
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgElement || !imgWrap) return;
    
    const imgRect = imgElement.getBoundingClientRect();
    const wrapRect = imgWrap.getBoundingClientRect();
    
    // Вычисляем относительные координаты
    const relX = clientX - wrapRect.left;
    const relY = clientY - wrapRect.top;
    
    const bbox = rect.bbox || {};
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;
    
    currentState.dragState = {
      rectIndex: rectIndex,
      startX: relX,
      startY: relY,
      originalX: (bbox.x || 0) * scaleX,
      originalY: (bbox.y || 0) * scaleY,
      originalW: (bbox.w || 0) * scaleX,
      originalH: (bbox.h || 0) * scaleY,
      scaleX: scaleX,
      scaleY: scaleY
    };
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', endDrag);
  }

  /**
   * Обрабатывает перемещение при drag&drop
   */
  function handleDrag(e) {
    if (!currentState.dragState) return;
    
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    const wrapRect = imgWrap.getBoundingClientRect();
    const relX = e.clientX - wrapRect.left;
    const relY = e.clientY - wrapRect.top;
    
    const dragState = currentState.dragState;
    const deltaX = relX - dragState.startX;
    const deltaY = relY - dragState.startY;
    
    const newX = dragState.originalX + deltaX;
    const newY = dragState.originalY + deltaY;
    
    // Обновляем позицию rectangle на экране
    const rectElement = imgWrap.querySelector(`[data-rect-index="${dragState.rectIndex}"]`);
    if (rectElement) {
      rectElement.style.left = `${newX}px`;
      rectElement.style.top = `${newY}px`;
    }
  }

  /**
   * Завершает drag&drop и сохраняет изменения
   */
  async function endDrag(e) {
    if (!currentState.dragState) return;
    
    document.removeEventListener('mousemove', handleDrag);
    document.removeEventListener('mouseup', endDrag);
    
    const dragState = currentState.dragState;
    const imgElement = document.getElementById('photoCardImg');
    const imgWrap = document.getElementById('photoCardImgWrap');
    
    if (!imgElement || !imgWrap) {
      currentState.dragState = null;
      return;
    }
    
    const rectElement = imgWrap.querySelector(`[data-rect-index="${dragState.rectIndex}"]`);
    if (!rectElement) {
      currentState.dragState = null;
      return;
    }
    
    // Вычисляем новые координаты в натуральных размерах изображения
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgRect = imgElement.getBoundingClientRect();
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    
    const displayX = parseFloat(rectElement.style.left);
    const displayY = parseFloat(rectElement.style.top);
    const displayW = parseFloat(rectElement.style.width);
    const displayH = parseFloat(rectElement.style.height);
    
    const naturalX = Math.round((displayX / imgDisplayWidth) * imgNaturalWidth);
    const naturalY = Math.round((displayY / imgDisplayHeight) * imgNaturalHeight);
    const naturalW = Math.round((displayW / imgDisplayWidth) * imgNaturalWidth);
    const naturalH = Math.round((displayH / imgDisplayHeight) * imgNaturalHeight);
    
    // Обновляем rectangle через API
    const rect = currentState.rectangles[dragState.rectIndex];
    if (rect && rect.id) {
      // Сохраняем старое состояние для UNDO (поддерживаем оба формата)
      let oldX, oldY, oldW, oldH;
      if (rect.bbox && typeof rect.bbox === 'object') {
        oldX = rect.bbox.x || 0;
        oldY = rect.bbox.y || 0;
        oldW = rect.bbox.w || 0;
        oldH = rect.bbox.h || 0;
      } else {
        oldX = rect.bbox_x || 0;
        oldY = rect.bbox_y || 0;
        oldW = rect.bbox_w || 0;
        oldH = rect.bbox_h || 0;
      }
      const oldBbox = { x: oldX, y: oldY, w: oldW, h: oldH };
      
      try {
        await updateRectangle(rect.id, {
          x: naturalX,
          y: naturalY,
          w: naturalW,
          h: naturalH
        });
        
        // Сохраняем действие в стек UNDO
        pushUndoAction({
          type: 'update_rectangle',
          face_rectangle_id: rect.id,
          oldBbox: oldBbox,
          newBbox: { x: naturalX, y: naturalY, w: naturalW, h: naturalH },
          oldPersonId: rect.person_id || null,
          oldAssignmentType: null
        });
        
        updateUndoButton();
        
        // Перезагружаем rectangles для синхронизации
        await loadRectangles();
      } catch (error) {
        console.error('[photo_card] Error updating rectangle:', error);
        // Восстанавливаем исходное положение
        drawRectangles();
      }
    }
    
    currentState.dragState = null;
  }

  /**
   * Обновляет rectangle через API
   */
  async function updateRectangle(faceRectangleId, bbox, personId, assignmentType) {
    if (!currentState.pipeline_run_id) {
      throw new Error('pipeline_run_id is required');
    }
    
    const payload = {
      pipeline_run_id: currentState.pipeline_run_id,
      face_rectangle_id: faceRectangleId
    };
    
    if (bbox) {
      payload.bbox = bbox;
    }
    
    if (personId !== undefined) {
      payload.person_id = personId;
    }
    
    if (assignmentType) {
      payload.assignment_type = assignmentType;
    }
    
    const response = await fetch('/api/faces/rectangle/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to update rectangle');
    }
    
    return await response.json();
  }

  /**
   * Удаляет rectangle
   */
  async function deleteRectangle(faceRectangleId) {
    if (!currentState.pipeline_run_id) {
      throw new Error('pipeline_run_id is required');
    }
    
    // Сохраняем старое состояние для UNDO
    const rect = currentState.rectangles.find(r => r.id === faceRectangleId);
    if (rect) {
      pushUndoAction({
        type: 'delete_rectangle',
        face_rectangle_id: faceRectangleId,
        oldBbox: rect.bbox || {},
        oldPersonId: rect.person_id || null,
        oldAssignmentType: null
      });
    }
    
    const response = await fetch('/api/faces/rectangle/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pipeline_run_id: currentState.pipeline_run_id,
        face_rectangle_id: faceRectangleId
      })
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Failed to delete rectangle');
    }
    
    updateUndoButton();
    
    // Перезагружаем rectangles
    await loadRectangles();
    await checkDuplicates();
  }

  /**
   * Проверяет дубликаты персоны на фото
   */
  async function checkDuplicates() {
    // Проверка дубликатов работает только для sorting режима (с pipeline_run_id)
    if (!currentState.pipeline_run_id || currentState.mode !== 'sorting') {
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
      
      const response = await fetch(`/api/faces/rectangles/duplicates-check?${params.toString()}`);
      if (!response.ok) return;
      
      const data = await response.json();
      if (data.ok && data.rectangles) {
        // Обновляем флаг is_duplicate для каждого rectangle
        const duplicateMap = {};
        data.rectangles.forEach(rect => {
          if (rect.is_duplicate) {
            duplicateMap[rect.face_rectangle_id] = true;
          }
        });
        
        currentState.rectangles.forEach(rect => {
          rect.is_duplicate = duplicateMap[rect.id] || false;
        });
        
        drawRectangles();
        updateRectanglesList();
      }
    } catch (error) {
      console.error('[photo_card] Error checking duplicates:', error);
    }
  }

  /**
   * Загружает список персон для модального окна
   */
  async function loadPersons() {
    try {
      const response = await fetch('/api/persons/list');
      if (!response.ok) return;
      
      const data = await response.json();
      if (data.persons) {
        currentState.allPersons = data.persons;
      }
    } catch (error) {
      console.error('[photo_card] Error loading persons:', error);
    }
  }

  /**
   * Показывает модальное окно выбора персоны
   */
  async function showPersonDialog(rectIndex) {
    currentState.editingRectangleIndex = rectIndex;
    
    // Загружаем список персон
    await loadPersons();
    
    // Заполняем select
    const select = document.getElementById('photoCardSelectPerson');
    if (select) {
      select.innerHTML = '<option value="">Выберите персону...</option>';
      currentState.allPersons.forEach(person => {
        const option = document.createElement('option');
        option.value = person.id;
        option.textContent = person.name + (person.is_me ? ' (я)' : '');
        select.appendChild(option);
      });
    }
    
    // Очищаем поле ввода
    const input = document.getElementById('photoCardInputNewPersonName');
    if (input) {
      input.value = '';
    }
    
    // Показываем модальное окно
    const modal = document.getElementById('photoCardPersonModal');
    if (modal) {
      modal.style.display = 'flex';
      modal.setAttribute('aria-hidden', 'false');
    }
  }

  /**
   * Закрывает модальное окно выбора персоны
   */
  function closePersonDialog() {
    const modal = document.getElementById('photoCardPersonModal');
    if (modal) {
      modal.style.display = 'none';
      modal.setAttribute('aria-hidden', 'true');
    }
    currentState.editingRectangleIndex = null;
  }

  /**
   * Назначает персону rectangle
   */
  async function assignPersonToRectangle() {
    const rectIndex = currentState.editingRectangleIndex;
    if (rectIndex === null) return;
    
    const rect = currentState.rectangles[rectIndex];
    if (!rect || !rect.id) return;
    
    const select = document.getElementById('photoCardSelectPerson');
    const input = document.getElementById('photoCardInputNewPersonName');
    
    let personId = null;
    const selectedValue = select ? select.value : '';
    const newPersonName = input ? input.value.trim() : '';
    
    if (selectedValue) {
      personId = parseInt(selectedValue);
    } else if (newPersonName) {
      // Создаем новую персону
      try {
        const response = await fetch('/api/persons/create', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: newPersonName })
        });
        if (response.ok) {
          const data = await response.json();
          if (data.person_id) {
            personId = data.person_id;
            // Обновляем список персон
            await loadPersons();
          }
        } else {
          const error = await response.json();
          throw new Error(error.detail || 'Failed to create person');
        }
      } catch (error) {
        console.error('[photo_card] Error creating person:', error);
        alert('Ошибка при создании персоны: ' + error.message);
        return;
      }
    }
    
    if (!personId) {
      alert('Выберите персону или введите имя новой персоны');
      return;
    }
    
    // Сохраняем старое состояние для UNDO
    const oldPersonId = rect.person_id || null;
    
    // Назначаем персону через API
    try {
      await updateRectangle(rect.id, null, personId, 'manual_face');
      
      // Сохраняем действие в стек UNDO
      pushUndoAction({
        type: 'assign_person',
        face_rectangle_id: rect.id,
        oldPersonId: oldPersonId,
        newPersonId: personId,
        oldAssignmentType: null,
        newAssignmentType: 'manual_face'
      });
      
      updateUndoButton();
      closePersonDialog();
      // Перезагружаем rectangles
      await loadRectangles();
      await checkDuplicates();
    } catch (error) {
      console.error('[photo_card] Error assigning person:', error);
      alert('Ошибка при назначении персоны: ' + error.message);
    }
  }
  
  /**
   * Обновляет видимость кнопки UNDO
   */
  function updateUndoButton() {
    const undoBtn = document.getElementById('photoCardUndo');
    if (undoBtn) {
      undoBtn.style.display = undoStack.length > 0 ? 'inline-block' : 'none';
    }
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
        toggleBtn.textContent = currentState.showRectangles ? 'Скрыть прямоугольники' : 'Показать прямоугольники';
        if (currentState.showRectangles) {
          drawRectangles();
        } else {
          const oldRects = document.querySelectorAll('.photo-card-rectangle');
          oldRects.forEach(el => el.remove());
        }
      });
    }
    
    // Модальное окно выбора персоны
    const personModalClose = document.getElementById('photoCardPersonModalClose');
    const personModalCancel = document.getElementById('photoCardPersonModalCancel');
    const personModalAssign = document.getElementById('photoCardAssignPerson');
    
    if (personModalClose) {
      personModalClose.addEventListener('click', closePersonDialog);
    }
    if (personModalCancel) {
      personModalCancel.addEventListener('click', closePersonDialog);
    }
    if (personModalAssign) {
      personModalAssign.addEventListener('click', assignPersonToRectangle);
    }
    
    // Закрытие модального окна по клику вне его
    const personModal = document.getElementById('photoCardPersonModal');
    if (personModal) {
      personModal.addEventListener('click', function(e) {
        if (e.target === personModal) {
          closePersonDialog();
        }
      });
    }
    
    // Специальные действия
    const assignOutsiderBtn = document.getElementById('photoCardAssignOutsider');
    const markAsCatBtn = document.getElementById('photoCardMarkAsCat');
    const markAsNoPeopleBtn = document.getElementById('photoCardMarkAsNoPeople');
    
    if (assignOutsiderBtn) {
      assignOutsiderBtn.addEventListener('click', async function() {
        if (confirm('Назначить все неназначенные rectangles персоне "Посторонний"?')) {
          try {
            const response = await fetch('/api/faces/rectangles/assign-outsider', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                pipeline_run_id: currentState.pipeline_run_id,
                file_id: currentState.file_id,
                path: currentState.file_path
              })
            });
            if (response.ok) {
              await loadRectangles();
              await checkDuplicates();
            }
          } catch (error) {
            console.error('[photo_card] Error assigning outsider:', error);
            alert('Ошибка: ' + error.message);
          }
        }
      });
    }
    
    if (markAsCatBtn) {
      markAsCatBtn.addEventListener('click', async function() {
        if (confirm('Пометить файл как "кот"? Все rectangles будут удалены.')) {
          try {
            const response = await fetch('/api/faces/file/mark-as-cat', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                pipeline_run_id: currentState.pipeline_run_id,
                file_id: currentState.file_id,
                path: currentState.file_path
              })
            });
            if (response.ok) {
              await loadRectangles();
              alert('Файл помечен как "кот"');
            }
          } catch (error) {
            console.error('[photo_card] Error marking as cat:', error);
            alert('Ошибка: ' + error.message);
          }
        }
      });
    }
    
    if (markAsNoPeopleBtn) {
      markAsNoPeopleBtn.addEventListener('click', async function() {
        if (confirm('Пометить файл как "нет людей"? Все rectangles будут удалены.')) {
          try {
            const response = await fetch('/api/faces/file/mark-as-no-people', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                pipeline_run_id: currentState.pipeline_run_id,
                file_id: currentState.file_id,
                path: currentState.file_path
              })
            });
            if (response.ok) {
              await loadRectangles();
              alert('Файл помечен как "нет людей"');
            }
          } catch (error) {
            console.error('[photo_card] Error marking as no people:', error);
            alert('Ошибка: ' + error.message);
          }
        }
      });
    }
    
    // Навигация
    const prevBtn = document.getElementById('photoCardPrev');
    const nextBtn = document.getElementById('photoCardNext');
    
    if (prevBtn) {
      prevBtn.addEventListener('click', navigatePrev);
    }
    if (nextBtn) {
      nextBtn.addEventListener('click', navigateNext);
    }
    
    // Навигация клавиатурой (стрелки влево/вправо)
    document.addEventListener('keydown', function(e) {
      const modal = document.getElementById('photoCardModal');
      if (!modal || modal.style.display === 'none') return;
      
      // Игнорируем, если фокус в input/textarea
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        navigatePrev();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        navigateNext();
      }
    });
    
    // Кнопка UNDO
    const undoBtn = document.getElementById('photoCardUndo');
    if (undoBtn) {
      undoBtn.addEventListener('click', undoLastAction);
    }
  }
  
  // UNDO система
  let undoStack = [];
  const MAX_UNDO_STACK_SIZE = 20;
  
  /**
   * Добавляет действие в стек UNDO
   */
  function pushUndoAction(action) {
    undoStack.push(action);
    if (undoStack.length > MAX_UNDO_STACK_SIZE) {
      undoStack.shift(); // Удаляем самое старое действие
    }
  }
  
  /**
   * Отменяет последнее действие
   */
  async function undoLastAction() {
    if (undoStack.length === 0) {
      alert('Нет действий для отмены');
      return;
    }
    
    const action = undoStack.pop();
    
    try {
      switch (action.type) {
        case 'update_rectangle':
          // Восстанавливаем предыдущее состояние rectangle
          await updateRectangle(action.face_rectangle_id, action.oldBbox, action.oldPersonId, action.oldAssignmentType);
          await loadRectangles();
          await checkDuplicates();
          break;
        case 'delete_rectangle':
          // Восстанавливаем rectangle (нужен API для восстановления)
          alert('Восстановление удаленного rectangle пока не реализовано');
          break;
        case 'assign_person':
          // Удаляем привязку персоны
          await updateRectangle(action.face_rectangle_id, null, null, null);
          await loadRectangles();
          await checkDuplicates();
          break;
        default:
          console.warn('[photo_card] Unknown undo action type:', action.type);
      }
    } catch (error) {
      console.error('[photo_card] Error undoing action:', error);
      alert('Ошибка при отмене действия: ' + error.message);
      // Возвращаем действие в стек, если отмена не удалась
      undoStack.push(action);
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
