/**
 * Единый модуль карточки фотографий для отображения и редактирования фото с rectangles
 * Используется на всех страницах проекта (faces.html, person_detail.html, face_cluster_detail.html)
 */

/**
 * Отправляет лог на сервер для записи в файл
 */
async function logToServer(level, message, data = null) {
  try {
    await fetch('/api/debug/client-log', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, message, data })
    });
  } catch (error) {
    // Игнорируем ошибки отправки логов
  }
}

// Обертка для console.log с записью на сервер
const originalConsoleLog = console.log;
const originalConsoleWarn = console.warn;
const originalConsoleError = console.error;

console.log = function(...args) {
  originalConsoleLog.apply(console, args);
  if (args[0] && typeof args[0] === 'string' && args[0].includes('[photo_card]')) {
    const message = args.join(' ');
    const data = args.length > 1 && typeof args[args.length - 1] === 'object' ? args[args.length - 1] : null;
    logToServer('log', message, data);
  }
};

console.warn = function(...args) {
  originalConsoleWarn.apply(console, args);
  if (args[0] && typeof args[0] === 'string' && args[0].includes('[photo_card]')) {
    const message = args.join(' ');
    const data = args.length > 1 && typeof args[args.length - 1] === 'object' ? args[args.length - 1] : null;
    logToServer('warn', message, data);
  }
};

console.error = function(...args) {
  originalConsoleError.apply(console, args);
  if (args[0] && typeof args[0] === 'string' && args[0].includes('[photo_card]')) {
    const message = args.join(' ');
    const data = args.length > 1 && typeof args[args.length - 1] === 'object' ? args[args.length - 1] : null;
    logToServer('error', message, data);
  }
};

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
    isEditMode: false, // Режим редактирования (показ якорей для перемещения и изменения размера)
    dragState: null, // Состояние drag&drop/resize: {rectIndex, anchorType, startX, startY, originalX, originalY, originalW, originalH}
    allPersons: [], // Список всех персон для модального окна
    editingRectangleIndex: null, // Индекс rectangle, для которого открыто модальное окно выбора персоны
    drawingState: null // Состояние рисования: {startX, startY, currentX, currentY, tempRectElement}
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
      isEditMode: false,
      dragState: null,
      allPersons: [],
      editingRectangleIndex: null,
      drawingState: null
    };

    // Если передан list_context, берем file_id, file_path и pipeline_run_id из него
    if (currentState.list_context && currentState.list_context.items) {
      const currentIndex = currentState.list_context.current_index || 0;
      const currentItem = currentState.list_context.items[currentIndex];
      if (currentItem) {
        currentState.file_id = currentItem.file_id || currentState.file_id;
        currentState.file_path = currentItem.file_path || currentState.file_path;
        // Берем pipeline_run_id из текущего элемента, если он есть
        if (currentItem.pipeline_run_id && !currentState.pipeline_run_id) {
          currentState.pipeline_run_id = currentItem.pipeline_run_id;
        }
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
    
    // Если pipeline_run_id всё ещё нет, но есть run_id в list_context, предупреждаем
    if (!currentState.pipeline_run_id && currentState.list_context && currentState.list_context.items) {
      const currentItem = currentState.list_context.items[currentState.list_context.current_index || 0];
      if (currentItem && (currentItem.run_id || currentItem.cluster_run_id)) {
        console.warn('[photo_card] Has run_id but no pipeline_run_id, rectangles may not load for sorting mode');
        console.warn('[photo_card] Current item:', currentItem);
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
    
    // Показываем навигацию (она всегда видна, но кнопки навигации показываются только при наличии list_context)
    const navigation = document.getElementById('photoCardNavigation');
    if (navigation) {
      navigation.style.display = 'flex';
      // Показываем кнопки навигации только если есть list_context
      const prevBtn = document.getElementById('photoCardPrev');
      const nextBtn = document.getElementById('photoCardNext');
      const positionElement = document.getElementById('photoCardPosition');
      if (currentState.list_context) {
        if (prevBtn) prevBtn.style.display = '';
        if (nextBtn) nextBtn.style.display = '';
        if (positionElement) positionElement.style.display = '';
        updateNavigationPosition();
      } else {
        if (prevBtn) prevBtn.style.display = 'none';
        if (nextBtn) nextBtn.style.display = 'none';
        if (positionElement) positionElement.style.display = 'none';
      }
    }

    // Показываем модальное окно
    const modal = document.getElementById('photoCardModal');
    if (!modal) {
      console.error('[photo_card] Modal element not found. Make sure photo_card.html is included.');
      return;
    }

    // ПОЛНАЯ ОЧИСТКА перед открытием - удаляем все старые элементы
    clearPhotoCardUI();

    modal.style.display = 'flex';
    modal.setAttribute('aria-hidden', 'false');

    // Загружаем данные
    loadPhotoCardData();
  }

  /**
   * Полностью очищает UI карточки (прямоугольники, меню, плашки)
   */
  function clearPhotoCardUI() {
    // Очищаем прямоугольники и меню на изображении
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (imgWrap) {
      // Удаляем все прямоугольники
      const oldRects = imgWrap.querySelectorAll('.photo-card-rectangle');
      oldRects.forEach(el => el.remove());
      
      // Удаляем все меню на изображении
      const menus = imgWrap.querySelectorAll('.photo-card-rectangle-menu');
      menus.forEach(el => el.remove());
      
      // Удаляем все якоря редактирования
      const anchors = imgWrap.querySelectorAll('.photo-card-rectangle-anchor');
      anchors.forEach(el => el.remove());
    }
    
    // Очищаем плашки со списком rectangles
    const rectList = document.getElementById('photoCardRectList');
    if (rectList) {
      rectList.innerHTML = '';
    }
    
    // Закрываем все открытые меню на плашках
    document.querySelectorAll('.rectpill-actions-menu.open').forEach(menu => {
      menu.classList.remove('open');
    });
    
    // Очищаем состояние
    currentState.selectedRectangleIndex = null;
    currentState.rectangles = [];
    currentState.isEditMode = false;
    currentState.dragState = null;
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
    
    // Очищаем rectangles при закрытии
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (imgWrap) {
      const oldRects = imgWrap.querySelectorAll('.photo-card-rectangle');
      oldRects.forEach(el => el.remove());
      // Также удаляем меню, если есть
      const menus = imgWrap.querySelectorAll('.photo-card-rectangle-menu');
      menus.forEach(el => el.remove());
    }
    
    // Очищаем состояние
    currentState.selectedRectangleIndex = null;
    currentState.rectangles = [];
    
    // Сохраняем контекст для возврата (если есть list_context)
    if (currentState.list_context) {
      const sourcePage = currentState.list_context.source_page;
      const contextKey = `photoCard_context_${sourcePage}`;
      
      try {
        // Сохраняем обновленный контекст с текущим индексом
        const contextData = {
          current_index: currentState.list_context.current_index,
          total_count: currentState.list_context.total_count,
          file_id: currentState.file_id,
          file_path: currentState.file_path,
          // Сохраняем параметры для восстановления вкладок/подвкладок, если они есть
          tab: currentState.list_context.tab || null,
          subtab: currentState.list_context.subtab || null
        };
        
        // Для faces.html нужно также сохранить параметры фильтрации
        if (sourcePage === 'faces' && currentState.list_context.api_fallback) {
          const apiParams = currentState.list_context.api_fallback.params || {};
          contextData.api_params = apiParams;
        }
        
        sessionStorage.setItem(contextKey, JSON.stringify(contextData));
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
      editingRectangleIndex: null,
      drawingState: null
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
      
      // Добавляем обработчики рисования
      attachDrawingHandlers();
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
        console.log('[photo_card] Image loaded, drawing rectangles if available');
        if (currentState.showRectangles && currentState.rectangles.length > 0) {
          // Небольшая задержка для гарантии, что размеры изображения обновлены
          setTimeout(() => {
            drawRectangles();
          }, 100);
        }
      };
      
      // Если изображение уже загружено, рисуем сразу
      if (imgElement.complete && imgElement.naturalWidth > 0 && imgElement.naturalHeight > 0) {
        console.log('[photo_card] Image already complete, drawing rectangles if available');
        if (currentState.showRectangles && currentState.rectangles.length > 0) {
          // Небольшая задержка для гарантии, что размеры изображения обновлены
          setTimeout(() => {
            drawRectangles();
          }, 100);
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
      // pipeline_run_id опционален (для архивных фотографий), но обязателен для сортируемых
      if (currentState.pipeline_run_id) {
        params.append('pipeline_run_id', currentState.pipeline_run_id);
      } else if (currentState.mode === 'sorting') {
        console.warn('[photo_card] pipeline_run_id is missing for sorting mode, rectangles may not load');
      }
      if (currentState.file_id) {
        params.append('file_id', currentState.file_id);
      } else if (currentState.file_path) {
        params.append('path', currentState.file_path);
      }

      console.log('[photo_card] Loading rectangles with params:', {
        pipeline_run_id: currentState.pipeline_run_id,
        file_id: currentState.file_id,
        file_path: currentState.file_path,
        mode: currentState.mode,
        params: params.toString()
      });

      const response = await fetch(`/api/faces/rectangles?${params.toString()}`);
      if (!response.ok) {
        const errorText = await response.text();
        console.error('[photo_card] Failed to load rectangles:', response.status, errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('[photo_card] Loaded rectangles:', data.ok, data.rectangles?.length || 0, 'rectangles', {
        ok: data.ok,
        count: data.rectangles?.length || 0,
        has_pipeline_run_id: !!currentState.pipeline_run_id
      });
      if (data.ok && data.rectangles) {
        currentState.rectangles = data.rectangles;
        // Сохраняем размеры изображения и EXIF orientation из ответа, если есть
        if (data.image_width && data.image_height) {
          currentState.originalImageSize = {
            width: data.image_width,
            height: data.image_height,
            exif_orientation: data.exif_orientation || null
          };
        }
        console.log('[photo_card] Rectangles data sample:', currentState.rectangles.slice(0, 2).map(r => ({
          id: r.id,
          bbox_x: r.bbox_x,
          bbox_y: r.bbox_y,
          bbox_w: r.bbox_w,
          bbox_h: r.bbox_h,
          bbox: r.bbox,
          person_name: r.person_name
        })));
        // Всегда обновляем список плашек, даже если изображение еще не загружено
        updateRectanglesList();
        
        // ВСЕГДА вызываем drawRectangles() для удаления старых rectangles с экрана
        // Это важно, когда rectangles были помечены как ignore_flag = 1 (например, после "нет людей")
        const imgElement = document.getElementById('photoCardImg');
        if (imgElement && imgElement.complete && imgElement.naturalWidth > 0 && imgElement.naturalHeight > 0) {
          console.log('[photo_card] Image already loaded, drawing rectangles');
          // Небольшая задержка для гарантии, что DOM обновлен
          setTimeout(() => {
            drawRectangles();
          }, 50);
        } else {
          console.log('[photo_card] Image not ready yet, will draw on load');
          // Даже если изображение не готово, нужно удалить старые rectangles
          // (они могут остаться на экране после обновления данных)
          const imgWrap = document.getElementById('photoCardImgWrap');
          if (imgWrap) {
            const oldRects = imgWrap.querySelectorAll('.photo-card-rectangle');
            console.log('[photo_card] Removing old rectangles (image not ready):', oldRects.length);
            oldRects.forEach(el => el.remove());
          }
        }
      } else {
        console.warn('[photo_card] No rectangles in response:', data);
      }
    } catch (error) {
      console.error('[photo_card] Error loading rectangles:', error);
    }
  }

  /**
   * Конвертирует координаты bbox из originalSize в displaySize (для отрисовки)
   * @param {Object} bbox - {x, y, w, h} в координатах originalSize
   * @param {HTMLElement} imgElement - элемент изображения
   * @param {HTMLElement} imgWrap - контейнер изображения
   * @returns {Object} - {x, y, w, h} в координатах displaySize (относительно imgWrap)
   */
  function convertBboxOriginalToDisplay(bbox, imgElement, imgWrap) {
    if (!imgElement || !imgWrap) {
      console.error('[photo_card] convertBboxOriginalToDisplay: imgElement or imgWrap is null');
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    
    if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
      console.warn('[photo_card] convertBboxOriginalToDisplay: image not loaded yet', {
        naturalWidth: imgNaturalWidth,
        naturalHeight: imgNaturalHeight
      });
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    const imgRect = imgElement.getBoundingClientRect();
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    const wrapRect = imgWrap.getBoundingClientRect();
    
    if (imgDisplayWidth === 0 || imgDisplayHeight === 0) {
      console.warn('[photo_card] convertBboxOriginalToDisplay: image display size is zero', {
        displayWidth: imgDisplayWidth,
        displayHeight: imgDisplayHeight
      });
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    // Используем тот же способ вычисления смещения, что и в convertBboxDisplayToOriginal
    const imgOffsetX = imgRect.left - wrapRect.left;
    const imgOffsetY = imgRect.top - wrapRect.top;
    
    let bboxX = bbox.x || 0;
    let bboxY = bbox.y || 0;
    let bboxW = bbox.w || 0;
    let bboxH = bbox.h || 0;
    
    // Получаем исходные размеры изображения и EXIF orientation
    const originalImageSize = currentState.originalImageSize;
    
    if (originalImageSize && originalImageSize.width && originalImageSize.height) {
      const originalWidth = originalImageSize.width;
      const originalHeight = originalImageSize.height;
      const exifOrientation = originalImageSize.exif_orientation;
      const originalRatio = originalWidth / originalHeight;
      const previewRatio = imgNaturalWidth / imgNaturalHeight;
      const isRotated90 = Math.abs(originalRatio - 1/previewRatio) < 0.1;
      
      if (isRotated90 && exifOrientation) {
        const scaleX = imgNaturalWidth / originalHeight;
        const scaleY = imgNaturalHeight / originalWidth;
        
        if (exifOrientation === 6) {
          const tempX = bboxX;
          const tempW = bboxW;
          bboxX = bboxY * scaleX;
          bboxY = (originalWidth - tempX - tempW) * scaleY;
          bboxW = bboxH * scaleX;
          bboxH = tempW * scaleY;
        } else if (exifOrientation === 8) {
          const tempX = bboxX;
          const tempW = bboxW;
          bboxX = (originalHeight - bboxY - bboxH) * scaleX;
          bboxY = tempX * scaleY;
          bboxW = bboxH * scaleX;
          bboxH = tempW * scaleY;
        } else if (exifOrientation === 3) {
          const scaleX2 = imgNaturalWidth / originalWidth;
          const scaleY2 = imgNaturalHeight / originalHeight;
          bboxX = (originalWidth - bboxX - bboxW) * scaleX2;
          bboxY = (originalHeight - bboxY - bboxH) * scaleY2;
          bboxW = bboxW * scaleX2;
          bboxH = bboxH * scaleY2;
        } else {
          const scaleX2 = imgNaturalWidth / originalWidth;
          const scaleY2 = imgNaturalHeight / originalHeight;
          bboxX = bboxX * scaleX2;
          bboxY = bboxY * scaleY2;
          bboxW = bboxW * scaleX2;
          bboxH = bboxH * scaleY2;
        }
      } else {
        // Нет поворота на 90°, просто масштабируем от originalSize к naturalSize
        const scaleX2 = imgNaturalWidth / originalWidth;
        const scaleY2 = imgNaturalHeight / originalHeight;
        bboxX = bboxX * scaleX2;
        bboxY = bboxY * scaleY2;
        bboxW = bboxW * scaleX2;
        bboxH = bboxH * scaleY2;
      }
    } else if (bboxX >= imgNaturalWidth || bboxY >= imgNaturalHeight || (bboxX + bboxW) > imgNaturalWidth || (bboxY + bboxH) > imgNaturalHeight) {
      const originalWidth = Math.max(bboxX + bboxW, imgNaturalWidth);
      const originalHeight = Math.max(bboxY + bboxH, imgNaturalHeight);
      const scaleX = imgNaturalWidth / originalWidth;
      const scaleY = imgNaturalHeight / originalHeight;
      bboxX = bboxX * scaleX;
      bboxY = bboxY * scaleY;
      bboxW = bboxW * scaleX;
      bboxH = bboxH * scaleY;
    }
    
    // Ограничиваем границы
    bboxX = Math.max(0, Math.min(bboxX, imgNaturalWidth));
    bboxY = Math.max(0, Math.min(bboxY, imgNaturalHeight));
    if (bboxX + bboxW > imgNaturalWidth) {
      bboxW = Math.max(1, imgNaturalWidth - bboxX);
    }
    if (bboxY + bboxH > imgNaturalHeight) {
      bboxH = Math.max(1, imgNaturalHeight - bboxY);
    }
    bboxW = Math.max(1, Math.min(bboxW, imgNaturalWidth - bboxX));
    bboxH = Math.max(1, Math.min(bboxH, imgNaturalHeight - bboxY));
    
    // Масштабируем к отображаемому размеру
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;
    const x = bboxX * scaleX + imgOffsetX;
    const y = bboxY * scaleY + imgOffsetY;
    const w = bboxW * scaleX;
    const h = bboxH * scaleY;
    
    return { x, y, w, h };
  }
  
  /**
   * Конвертирует координаты bbox из displaySize в originalSize (для сохранения)
   * @param {Object} bbox - {x, y, w, h} в координатах displaySize (относительно imgWrap)
   * @param {HTMLElement} imgElement - элемент изображения
   * @param {HTMLElement} imgWrap - контейнер изображения
   * @returns {Object} - {x, y, w, h} в координатах originalSize
   */
  function convertBboxDisplayToOriginal(bbox, imgElement, imgWrap) {
    if (!imgElement || !imgWrap) {
      console.error('[photo_card] convertBboxDisplayToOriginal: imgElement or imgWrap is null');
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    
    if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
      console.warn('[photo_card] convertBboxDisplayToOriginal: image not loaded yet', {
        naturalWidth: imgNaturalWidth,
        naturalHeight: imgNaturalHeight
      });
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    const imgRect = imgElement.getBoundingClientRect();
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    const wrapRect = imgWrap.getBoundingClientRect();
    
    if (imgDisplayWidth === 0 || imgDisplayHeight === 0) {
      console.warn('[photo_card] convertBboxDisplayToOriginal: image display size is zero', {
        displayWidth: imgDisplayWidth,
        displayHeight: imgDisplayHeight
      });
      return { x: 0, y: 0, w: 0, h: 0 };
    }
    
    // Смещение изображения внутри wrap
    const imgOffsetX = imgRect.left - wrapRect.left;
    const imgOffsetY = imgRect.top - wrapRect.top;
    
    // Координаты относительно изображения (x и y уже относительно wrap)
    const imgX = bbox.x - imgOffsetX;
    const imgY = bbox.y - imgOffsetY;
    
    // Масштаб от отображаемого размера к натуральному
    const scaleX = imgNaturalWidth / imgDisplayWidth;
    const scaleY = imgNaturalHeight / imgDisplayHeight;
    
    // Координаты в натуральных размерах изображения (naturalSize)
    let bboxX = imgX * scaleX;
    let bboxY = imgY * scaleY;
    let bboxW = bbox.w * scaleX;
    let bboxH = bbox.h * scaleY;
    
    // Применяем обратную трансформацию EXIF (из naturalSize в originalSize)
    const originalImageSize = currentState.originalImageSize;
    if (originalImageSize && originalImageSize.width && originalImageSize.height) {
      const originalWidth = originalImageSize.width;
      const originalHeight = originalImageSize.height;
      const exifOrientation = originalImageSize.exif_orientation;
      const originalRatio = originalWidth / originalHeight;
      const previewRatio = imgNaturalWidth / imgNaturalHeight;
      const isRotated90 = Math.abs(originalRatio - 1/previewRatio) < 0.1;
      
      if (isRotated90 && exifOrientation) {
        // Обратная трансформация для поворота на 90°
        if (exifOrientation === 6) {
          const scaleX2 = imgNaturalWidth / originalHeight;
          const scaleY2 = imgNaturalHeight / originalWidth;
          const tempX = bboxX / scaleX2; // original bboxY
          const tempW = bboxW / scaleX2; // original bboxH
          const originalBboxY = (originalWidth - bboxY / scaleY2 - bboxH / scaleY2); // original bboxX
          const originalBboxH = bboxH / scaleY2; // original bboxW
          bboxX = originalBboxY;
          bboxY = tempX;
          bboxW = originalBboxH;
          bboxH = tempW;
        } else if (exifOrientation === 8) {
          const scaleX2 = imgNaturalWidth / originalHeight;
          const scaleY2 = imgNaturalHeight / originalWidth;
          const tempX = bboxY / scaleY2; // original bboxX
          const tempW = bboxH / scaleY2; // original bboxW
          const originalBboxY = (originalHeight - bboxX / scaleX2 - bboxW / scaleX2); // original bboxY
          const originalBboxH = bboxW / scaleX2; // original bboxH
          bboxX = tempX;
          bboxY = originalBboxY;
          bboxW = tempW;
          bboxH = originalBboxH;
        } else if (exifOrientation === 3) {
          const scaleX2 = originalWidth / imgNaturalWidth;
          const scaleY2 = originalHeight / imgNaturalHeight;
          bboxX = (originalWidth - bboxX / scaleX2 - bboxW / scaleX2);
          bboxY = (originalHeight - bboxY / scaleY2 - bboxH / scaleY2);
          bboxW = bboxW / scaleX2;
          bboxH = bboxH / scaleY2;
        }
      } else {
        // Нет поворота на 90°, просто масштабируем от naturalSize к originalSize
        const scaleX2 = originalWidth / imgNaturalWidth;
        const scaleY2 = originalHeight / imgNaturalHeight;
        bboxX = bboxX * scaleX2;
        bboxY = bboxY * scaleY2;
        bboxW = bboxW * scaleX2;
        bboxH = bboxH * scaleY2;
      }
    }
    
    return {
      x: Math.round(bboxX),
      y: Math.round(bboxY),
      w: Math.round(bboxW),
      h: Math.round(bboxH)
    };
  }

  /**
   * Отрисовывает rectangles на изображении
   */
  function drawRectangles() {
    const imgElement = document.getElementById('photoCardImg');
    const canvas = document.getElementById('photoCardCanvas');
    const imgWrap = document.getElementById('photoCardImgWrap');
    
    if (!imgElement || !canvas || !imgWrap) {
      console.warn('[photo_card] Missing DOM elements for drawing rectangles');
      return;
    }

    // ВСЕГДА очищаем старые rectangles (даже если новых нет)
    // Это важно, когда rectangles были удалены или помечены как ignore_flag = 1
    const oldRects = imgWrap.querySelectorAll('.photo-card-rectangle');
    console.log('[photo_card] Removing old rectangles:', oldRects.length);
    oldRects.forEach(el => el.remove());
    
    // Если rectangles скрыты или их нет, просто удаляем старые и выходим
    if (!currentState.showRectangles) {
      console.log('[photo_card] Rectangles hidden by showRectangles flag');
      return;
    }
    
    if (!currentState.rectangles || currentState.rectangles.length === 0) {
      console.log('[photo_card] No rectangles to draw (all removed)');
      return;
    }

    if (!imgElement.complete || imgElement.naturalWidth === 0 || imgElement.naturalHeight === 0) {
      console.warn('[photo_card] Image not ready, skipping draw:', {
        complete: imgElement.complete,
        naturalWidth: imgElement.naturalWidth,
        naturalHeight: imgElement.naturalHeight
      });
      return;
    }

    // Получаем размеры изображения
    const imgRect = imgElement.getBoundingClientRect();
    const imgWrapRect = imgWrap.getBoundingClientRect();
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    
    // Вычисляем смещение изображения относительно контейнера
    // ВАЖНО: offsetLeft/offsetTop дают смещение относительно родительского элемента (imgWrap)
    const imgOffsetX = imgElement.offsetLeft || 0;
    const imgOffsetY = imgElement.offsetTop || 0;

    console.log('[photo_card] drawRectangles called:', {
      rectanglesCount: currentState.rectangles.length,
      imageComplete: imgElement.complete,
      naturalSize: { w: imgNaturalWidth, h: imgNaturalHeight },
      displaySize: { w: imgDisplayWidth, h: imgDisplayHeight },
      imgRect: { width: imgRect.width, height: imgRect.height, left: imgRect.left, top: imgRect.top },
      imgWrapRect: { left: imgWrapRect.left, top: imgWrapRect.top, width: imgWrapRect.width, height: imgWrapRect.height },
      imgOffset: { x: imgOffsetX, y: imgOffsetY }
    });

    if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
      console.warn('[photo_card] Image not loaded yet, skipping draw:', { naturalWidth: imgNaturalWidth, naturalHeight: imgNaturalHeight });
      return;
    }

    if (imgDisplayWidth === 0 || imgDisplayHeight === 0) {
      console.warn('[photo_card] Image display size is zero, skipping draw:', { displayWidth: imgDisplayWidth, displayHeight: imgDisplayHeight });
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
      
      // Используем функцию конвертации координат
      const displayBbox = convertBboxOriginalToDisplay(
        { x: bbox_x, y: bbox_y, w: bbox_w, h: bbox_h },
        imgElement,
        imgWrap
      );
      const x = displayBbox.x;
      const y = displayBbox.y;
      const w = displayBbox.w;
      const h = displayBbox.h;
      
      console.log('[photo_card] Drawing rectangle', rect.id, {
        bbox: { x: bbox_x, y: bbox_y, w: bbox_w, h: bbox_h },
        display: { x, y, w, h },
        imageSize: { natural: { w: imgNaturalWidth, h: imgNaturalHeight }, display: { w: imgDisplayWidth, h: imgDisplayHeight } },
        originalImageSize: currentState.originalImageSize
      });

      // Проверяем дубликаты (красный восклицательный знак и красный текст)
      const isDuplicate = rect.is_duplicate || false;
      
      // Определяем цвет rectangle
      let color = 'rgba(250, 204, 21, 0.3)'; // Желтый по умолчанию (кластеры)
      let borderColor = 'rgba(250, 204, 21, 1)';
      let labelBackground = 'rgba(250, 204, 21, 0.95)'; // Желтый фон для подписи
      let labelColor = '#111827'; // Темный текст для желтого фона
      
      // Проверяем, это выделенный rectangle через highlight_rectangle?
      let isHighlighted = false;
      if (currentState.highlight_rectangle) {
        isHighlighted = 
          (currentState.highlight_rectangle.type === 'face_rectangle' && rect.id === currentState.highlight_rectangle.id) ||
          (currentState.highlight_rectangle.type === 'person_rectangle' && rect.person_rectangle_id === currentState.highlight_rectangle.id);
        
        if (isHighlighted) {
          color = 'rgba(11, 87, 208, 0.3)'; // Синий для выделенного
          borderColor = 'rgba(11, 87, 208, 1)';
          labelBackground = 'rgba(11, 87, 208, 0.95)'; // Синий фон для подписи
          labelColor = '#fff'; // Белый текст для синего фона
        }
      }

      // Проверяем, это выделенный rectangle через selectedRectangleIndex?
      const isSelected = currentState.selectedRectangleIndex === index;
      if (isSelected) {
        color = 'rgba(34, 197, 94, 0.3)'; // Зеленый для выделенного
        borderColor = 'rgba(34, 197, 94, 1)';
        labelBackground = 'rgba(34, 197, 94, 0.95)'; // Зеленый фон для подписи
        labelColor = '#fff'; // Белый текст для зеленого фона
      }
      
      // Для дубликатов - только красный текст, рамка остается как есть
      if (isDuplicate) {
        labelColor = '#dc2626'; // Красный текст для дубликатов
      }
      
      // Проверяем тип привязки для стиля границы (пунктир для ручных привязок)
      const isManualFace = rect.assignment_type === 'manual_face';
      const borderStyle = isManualFace ? 'dashed' : 'solid';
      
      // Создаем элемент rectangle
      const rectElement = document.createElement('div');
      rectElement.className = 'photo-card-rectangle';
      rectElement.style.position = 'absolute';
      rectElement.style.left = `${x}px`;
      rectElement.style.top = `${y}px`;
      rectElement.style.width = `${w}px`;
      rectElement.style.height = `${h}px`;
      rectElement.style.border = `2px ${borderStyle} ${borderColor}`;
      rectElement.style.backgroundColor = color;
      rectElement.style.pointerEvents = 'auto';
      rectElement.style.cursor = currentState.isEditMode ? 'default' : 'pointer';
      rectElement.style.zIndex = isSelected ? '1001' : '1000';
      if (currentState.isEditMode) {
        rectElement.classList.add('edit-mode');
        rectElement.title = 'Режим редактирования: используйте якоря для перемещения и изменения размера';
      } else {
        rectElement.title = 'Клик - меню действий';
      }
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
        label.style.background = labelBackground;
        label.style.color = labelColor;
        label.style.padding = '3px 8px';
        label.style.fontSize = '12px';
        label.style.fontWeight = '700';
        label.style.borderRadius = '4px';
        label.style.whiteSpace = 'nowrap';
        label.style.zIndex = '10002';
        label.style.pointerEvents = 'none';
        if (isDuplicate) {
          label.style.color = '#dc2626'; // Красный текст для дубликатов
        }
        rectElement.appendChild(label);
      }
      
      // Добавляем обработчики событий
      // Клик - выделение rectangle и показ выпадашки (только если не в режиме редактирования)
      rectElement.addEventListener('click', function(e) {
        e.stopPropagation();
        // Всегда выделяем rectangle при клике
        selectRectangle(index);
        if (!currentState.isEditMode) {
          // Показываем выпадашку при клике на rectangle
          showRectangleActionsMenuOnImage(index, e);
        }
      });
      
      // Двойной клик - переход к персоне (отложено на этап 3)
      // Пока оставляем назначение персоны как временное решение
      rectElement.addEventListener('dblclick', function(e) {
        e.stopPropagation();
        if (!currentState.isEditMode) {
          // TODO: Этап 3 - переход к персоне
          // Пока открываем диалог назначения персоны
          showPersonDialog(index);
        }
      });
      
      // В режиме редактирования показываем якоря
      if (currentState.isEditMode) {
        addEditAnchors(rectElement, index);
      }
      
      // Добавляем rectangle в контейнер изображения
      imgWrap.appendChild(rectElement);
      
      console.log('[photo_card] Rectangle element added to DOM:', {
        rectId: rect.id,
        element: rectElement,
        parent: imgWrap,
        position: { x, y, w, h },
        styles: {
          left: rectElement.style.left,
          top: rectElement.style.top,
          width: rectElement.style.width,
          height: rectElement.style.height
        }
      });
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
      // Добавляем разделитель перед каждым элементом, кроме первого
      if (index > 0) {
        const separator = document.createElement('span');
        separator.textContent = ' • ';
        separator.style.color = '#6b7280';
        separator.style.margin = '0 6px';
        separator.style.fontSize = '16px';
        separator.style.fontWeight = 'bold';
        separator.style.verticalAlign = 'middle';
        rectList.appendChild(separator);
        console.log('[photo_card] Added separator before rectangle', index);
      }
      
      const pill = document.createElement('div');
      pill.className = 'rectpill';
      pill.style.display = 'inline-flex';
      pill.style.alignItems = 'center';
      pill.style.gap = '8px';
      pill.style.cursor = 'pointer';
      pill.style.position = 'relative';
      pill.dataset.rectIndex = index;
      
      // Исходные стили для невыделенного rectangle
      const defaultBackground = '#eef2ff';
      const defaultBorder = '1px solid #c7d2fe';
      const selectedBackground = '#c7d2fe';
      const selectedBorder = '1px solid #6366f1';
      
      // Применяем исходные стили
      pill.style.background = defaultBackground;
      pill.style.border = defaultBorder;
      
      // Текст с именем персоны или номером
      const text = document.createElement('span');
      text.textContent = (rect.is_duplicate ? '⚠ ' : '') + (rect.person_name || `Rectangle ${index + 1}`);
      if (rect.is_duplicate) {
        text.style.color = '#dc2626'; // Красный текст для дубликатов в плашке
      }
      pill.appendChild(text);
      
      // Выделение активного rectangle
      if (currentState.selectedRectangleIndex === index) {
        pill.style.background = selectedBackground;
        pill.style.border = selectedBorder;
      }
      
      // Клик по плашке - выделение/снятие выделения rectangle
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
      }
      
      // Кнопка действий (выпадающее меню) - показываем всегда
      const actionsBtn = document.createElement('button');
      actionsBtn.className = 'rectpill-action';
      actionsBtn.textContent = '⋮';
      actionsBtn.style.background = 'none';
      actionsBtn.style.border = 'none';
      actionsBtn.style.cursor = 'pointer';
      actionsBtn.style.padding = '2px 6px';
      actionsBtn.style.fontSize = '16px';
      actionsBtn.style.lineHeight = '1';
      actionsBtn.style.color = '#6b7280';
      actionsBtn.title = 'Действия с прямоугольником';
      actionsBtn.addEventListener('click', function(e) {
        e.stopPropagation();
        showRectangleActionsMenu(index, e.target);
      });
      pill.appendChild(actionsBtn);

      rectList.appendChild(pill);
    });
  }

  /**
   * Создает компактное меню действий для rectangle
   */
  function createRectangleActionsMenu(rectIndex) {
    const rect = currentState.rectangles[rectIndex];
    if (!rect) return null;
    
    const menu = document.createElement('div');
    menu.className = 'rectpill-actions-menu';
    
    const hasPerson = rect.person_id !== null && rect.person_id !== undefined;
    
    // Кнопка "Посторонние" - отдельный пункт меню первого уровня (первым)
    const outsiderBtn = document.createElement('button');
    outsiderBtn.textContent = 'Посторонний';
    outsiderBtn.addEventListener('click', async function(e) {
      e.stopPropagation();
      menu.classList.remove('open');
      
      // Находим персону "Посторонний" по флагу is_ignored
      if (!currentState.allPersons || currentState.allPersons.length === 0) {
        await loadPersons();
      }
      
      const outsiderPerson = currentState.allPersons.find(p => p.is_ignored === true);
      if (!outsiderPerson) {
        alert('Персона "Посторонний" не найдена');
        return;
      }
      
      // Назначаем персону
      const rect = currentState.rectangles[rectIndex];
      if (!rect || !rect.id) return;
      
      try {
        const oldPersonId = rect.person_id || null;
        await updateRectangle(rect.id, null, outsiderPerson.id, 'manual_face');
        
        pushUndoAction({
          type: 'assign_person',
          rectangle_id: rect.id,
          oldPersonId: oldPersonId,
          newPersonId: outsiderPerson.id,
          oldAssignmentType: rect.assignment_type || null,
          newAssignmentType: 'manual_face'
        });
        
        updateUndoButton();
        await loadRectangles();
        await checkDuplicates();
      } catch (error) {
        console.error('[photo_card] Error assigning outsider:', error);
        alert('Ошибка при назначении персоны: ' + error.message);
      }
    });
    menu.appendChild(outsiderBtn);
    
    // Кнопка "Назначить персону" или "Изменить персону" - с выпадающим списком
    const personContainer = document.createElement('div');
    personContainer.style.position = 'relative';
    
    const personBtn = document.createElement('button');
    personBtn.textContent = hasPerson ? 'Изменить персону' : 'Назначить персону';
    personBtn.style.display = 'flex';
    personBtn.style.justifyContent = 'space-between';
    personBtn.style.alignItems = 'center';
    personBtn.innerHTML = `<span>${hasPerson ? 'Изменить персону' : 'Назначить персону'}</span><span style="margin-left: 8px;">▶</span>`;
    personBtn.addEventListener('mouseenter', function(e) {
      e.stopPropagation();
      showPersonSubmenu(personContainer, rectIndex);
    });
    personContainer.appendChild(personBtn);
    menu.appendChild(personContainer);
    
    if (hasPerson) {
      
      // Кнопка "Изменить тип привязки" - переключение между кластером и ручной привязкой
      // Показываем только если персона назначена и есть assignment_type
      if (rect.assignment_type) {
        const changeTypeBtn = document.createElement('button');
        changeTypeBtn.textContent = rect.assignment_type === 'cluster' 
          ? 'Изменить на ручную привязку' 
          : 'Изменить на кластер';
        changeTypeBtn.addEventListener('click', async function(e) {
          e.stopPropagation();
          menu.classList.remove('open');
          
          const newType = rect.assignment_type === 'cluster' ? 'manual_face' : 'cluster';
          
          try {
            // Сохраняем старое состояние для UNDO
            const oldAssignmentType = rect.assignment_type;
            const oldPersonId = rect.person_id;
            
            // Обновляем через API (передаем тот же person_id, но меняем assignment_type)
            await updateRectangle(rect.id, null, rect.person_id, newType);
            
            // Сохраняем действие в стек UNDO
            pushUndoAction({
              type: 'change_assignment_type',
              rectangle_id: rect.id,
              oldAssignmentType: oldAssignmentType,
              newAssignmentType: newType,
              person_id: rect.person_id
            });
            
            updateUndoButton();
            
            // Перезагружаем rectangles
            await loadRectangles();
          } catch (error) {
            console.error('[photo_card] Error changing assignment type:', error);
            alert('Ошибка при изменении типа привязки: ' + error.message);
          }
        });
        menu.appendChild(changeTypeBtn);
      }
    }
    
    // Кнопка "Режим редактирования" / "Выйти из режима редактирования"
    const editModeBtn = document.createElement('button');
    editModeBtn.textContent = currentState.isEditMode ? 'Выйти из режима редактирования' : 'Режим редактирования';
    editModeBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      menu.classList.remove('open');
      toggleEditMode();
    });
    menu.appendChild(editModeBtn);
    
    // Кнопка "Удалить rectangle"
    const deleteBtn = document.createElement('button');
    deleteBtn.textContent = 'Удалить rectangle';
    deleteBtn.className = 'danger';
    deleteBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      menu.classList.remove('open');
      deleteRectangle(rect.id);
    });
    menu.appendChild(deleteBtn);
    
    return menu;
  }
  
  /**
   * Показывает выпадающий список персон справа от пункта меню
   */
  async function showPersonSubmenu(container, rectIndex) {
    // Закрываем все открытые подменю
    document.querySelectorAll('.person-submenu').forEach(submenu => {
      submenu.remove();
    });
    
    // Загружаем список персон, если еще не загружен
    if (!currentState.allPersons || currentState.allPersons.length === 0) {
      await loadPersons();
    }
    
    // Группируем персон по группам (исключая "Посторонние")
    const personsByGroup = {};
    const noGroupPersons = [];
    
    currentState.allPersons.forEach(person => {
      // Исключаем "Посторонние" из списка
      if (person.is_ignored === true) {
        return;
      }
      
      const group = person.group || null;
      if (group) {
        if (!personsByGroup[group]) {
          personsByGroup[group] = [];
        }
        personsByGroup[group].push(person);
      } else {
        noGroupPersons.push(person);
      }
    });
    
    // Создаем подменю
    const submenu = document.createElement('div');
    submenu.className = 'person-submenu';
    submenu.style.position = 'absolute';
    submenu.style.left = '100%';
    submenu.style.top = '0';
    submenu.style.marginLeft = '4px';
    submenu.style.background = '#fff';
    submenu.style.border = '1px solid #e5e7eb';
    submenu.style.borderRadius = '8px';
    submenu.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    submenu.style.minWidth = '200px';
    submenu.style.maxHeight = '400px';
    submenu.style.overflowY = 'auto';
    submenu.style.zIndex = '10001';
    submenu.style.padding = '4px 0';
    
    // Добавляем персон по группам
    const sortedGroups = Object.keys(personsByGroup).sort();
    sortedGroups.forEach(groupName => {
      const groupLabel = document.createElement('div');
      groupLabel.className = 'menu-group-label';
      groupLabel.textContent = groupName;
      groupLabel.style.padding = '6px 14px 4px 14px';
      groupLabel.style.fontSize = '11px';
      groupLabel.style.fontWeight = '600';
      groupLabel.style.color = '#6b7280';
      groupLabel.style.textTransform = 'uppercase';
      groupLabel.style.letterSpacing = '0.5px';
      submenu.appendChild(groupLabel);
      
      personsByGroup[groupName].forEach(person => {
        const personBtn = document.createElement('button');
        personBtn.textContent = person.name + (person.is_me ? ' (я)' : '');
        personBtn.style.display = 'block';
        personBtn.style.width = '100%';
        personBtn.style.padding = '8px 14px';
        personBtn.style.textAlign = 'left';
        personBtn.style.background = 'none';
        personBtn.style.border = 'none';
        personBtn.style.color = '#111827';
        personBtn.style.cursor = 'pointer';
        personBtn.style.fontSize = '13px';
        personBtn.style.transition = 'background 0.15s';
        personBtn.addEventListener('click', async function(e) {
          e.stopPropagation();
          submenu.remove();
          const menu = container.closest('.rectpill-actions-menu');
          if (menu) {
            menu.classList.remove('open');
          }
          
          // Назначаем персону
          const rect = currentState.rectangles[rectIndex];
          if (!rect || !rect.id) return;
          
          try {
            const oldPersonId = rect.person_id || null;
            await updateRectangle(rect.id, null, person.id, 'manual_face');
            
            pushUndoAction({
              type: 'assign_person',
              rectangle_id: rect.id,
              oldPersonId: oldPersonId,
              newPersonId: person.id,
              oldAssignmentType: rect.assignment_type || null,
              newAssignmentType: 'manual_face'
            });
            
            updateUndoButton();
            await loadRectangles();
            await checkDuplicates();
          } catch (error) {
            console.error('[photo_card] Error assigning person:', error);
            alert('Ошибка при назначении персоны: ' + error.message);
          }
        });
        personBtn.addEventListener('mouseenter', function() {
          personBtn.style.background = '#f9fafb';
        });
        personBtn.addEventListener('mouseleave', function() {
          personBtn.style.background = 'none';
        });
        submenu.appendChild(personBtn);
      });
    });
    
    // Добавляем персон без группы
    if (noGroupPersons.length > 0) {
      if (sortedGroups.length > 0) {
        const separator = document.createElement('div');
        separator.style.height = '1px';
        separator.style.background = '#e5e7eb';
        separator.style.margin = '4px 0';
        submenu.appendChild(separator);
      }
      
      noGroupPersons.forEach(person => {
        const personBtn = document.createElement('button');
        personBtn.textContent = person.name + (person.is_me ? ' (я)' : '');
        personBtn.style.display = 'block';
        personBtn.style.width = '100%';
        personBtn.style.padding = '8px 14px';
        personBtn.style.textAlign = 'left';
        personBtn.style.background = 'none';
        personBtn.style.border = 'none';
        personBtn.style.color = '#111827';
        personBtn.style.cursor = 'pointer';
        personBtn.style.fontSize = '13px';
        personBtn.style.transition = 'background 0.15s';
        personBtn.addEventListener('click', async function(e) {
          e.stopPropagation();
          submenu.remove();
          const menu = container.closest('.rectpill-actions-menu');
          if (menu) {
            menu.classList.remove('open');
          }
          
          // Назначаем персону
          const rect = currentState.rectangles[rectIndex];
          if (!rect || !rect.id) return;
          
          try {
            const oldPersonId = rect.person_id || null;
            await updateRectangle(rect.id, null, person.id, 'manual_face');
            
            pushUndoAction({
              type: 'assign_person',
              rectangle_id: rect.id,
              oldPersonId: oldPersonId,
              newPersonId: person.id,
              oldAssignmentType: rect.assignment_type || null,
              newAssignmentType: 'manual_face'
            });
            
            updateUndoButton();
            await loadRectangles();
            await checkDuplicates();
          } catch (error) {
            console.error('[photo_card] Error assigning person:', error);
            alert('Ошибка при назначении персоны: ' + error.message);
          }
        });
        personBtn.addEventListener('mouseenter', function() {
          personBtn.style.background = '#f9fafb';
        });
        personBtn.addEventListener('mouseleave', function() {
          personBtn.style.background = 'none';
        });
        submenu.appendChild(personBtn);
      });
    }
    
    container.appendChild(submenu);
    
    // Закрываем подменю при клике вне его
    const closeSubmenu = function(e) {
      if (!submenu.contains(e.target) && !container.contains(e.target)) {
        submenu.remove();
        document.removeEventListener('click', closeSubmenu);
      }
    };
    setTimeout(() => {
      document.addEventListener('click', closeSubmenu);
    }, 0);
  }
  
  /**
   * Показывает меню действий для rectangle (на плашке)
   */
  function showRectangleActionsMenu(rectIndex, buttonElement) {
    const rect = currentState.rectangles[rectIndex];
    if (!rect || !rect.id) return;
    
    // Закрываем все открытые меню
    document.querySelectorAll('.rectpill-actions-menu.open, .photo-card-rectangle-menu.open').forEach(menu => {
      menu.classList.remove('open');
    });
    
    // Проверяем, есть ли уже меню для этого rectangle
    let menu = buttonElement.closest('.rectpill')?.querySelector('.rectpill-actions-menu');
    
    if (!menu) {
      // Создаем меню
      menu = createRectangleActionsMenu(rectIndex);
      
      // Добавляем меню в плашку
      const pill = buttonElement.closest('.rectpill');
      if (pill) {
        pill.style.position = 'relative';
        pill.appendChild(menu);
      }
    }
    
    // Переключаем видимость меню
    menu.classList.toggle('open');
    
    // Закрываем меню при клике вне его
    const closeMenu = (e) => {
      if (!menu.contains(e.target) && !buttonElement.contains(e.target)) {
        menu.classList.remove('open');
        document.removeEventListener('click', closeMenu);
      }
    };
    
    // Добавляем обработчик закрытия с небольшой задержкой, чтобы не закрыть сразу
    setTimeout(() => {
      document.addEventListener('click', closeMenu);
    }, 10);
  }
  
  /**
   * Показывает меню действий для rectangle (на изображении)
   */
  function showRectangleActionsMenuOnImage(rectIndex, event) {
    const rect = currentState.rectangles[rectIndex];
    if (!rect) return;
    
    // Выделяем rectangle (если еще не выделен)
    if (currentState.selectedRectangleIndex !== rectIndex) {
      currentState.selectedRectangleIndex = rectIndex;
      drawRectangles();
      updateRectanglesList();
    }
    
    // Закрываем все открытые меню
    document.querySelectorAll('.rectpill-actions-menu.open, .photo-card-rectangle-menu').forEach(menu => {
      menu.remove();
    });
    
    // Проверяем, не открыто ли уже меню для этого rectangle
    const existingMenu = document.querySelector(`.photo-card-rectangle-menu[data-rect-index="${rectIndex}"]`);
    if (existingMenu) {
      existingMenu.remove();
      return; // Закрываем меню, если оно уже было открыто
    }
    
    // Создаем меню
    const menu = createRectangleActionsMenu(rectIndex);
    menu.className = 'photo-card-rectangle-menu';
    menu.classList.add('open');
    menu.dataset.rectIndex = rectIndex;
    
    // Позиционируем меню рядом с местом клика
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (imgWrap) {
      imgWrap.style.position = 'relative';
      const rectBounds = event.target.getBoundingClientRect();
      const wrapRect = imgWrap.getBoundingClientRect();
      menu.style.position = 'absolute';
      menu.style.left = `${rectBounds.left - wrapRect.left}px`;
      menu.style.top = `${rectBounds.top - wrapRect.top - menu.offsetHeight - 4}px`; // Вверх от rectangle
      imgWrap.appendChild(menu);
    }
    
    // Закрываем меню при клике вне его
    const closeMenu = (e) => {
      if (!menu.contains(e.target) && !event.target.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('click', closeMenu);
    }, 10);
  }


  /**
   * Переключает режим редактирования
   */
  function toggleEditMode() {
    currentState.isEditMode = !currentState.isEditMode;
    // Перерисовываем rectangles с якорями
    drawRectangles();
  }
  
  /**
   * Добавляет якоря для редактирования rectangle
   */
  function addEditAnchors(rectElement, rectIndex) {
    // Удаляем старые якоря, если есть
    const oldAnchors = rectElement.querySelectorAll('.photo-card-rectangle-anchor');
    oldAnchors.forEach(anchor => anchor.remove());
    
    // Типы якорей: углы (nw, ne, se, sw), стороны (n, e, s, w), центр (move)
    const anchorTypes = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'move'];
    
    anchorTypes.forEach(anchorType => {
      const anchor = document.createElement('div');
      anchor.className = `photo-card-rectangle-anchor ${anchorType}`;
      anchor.dataset.rectIndex = rectIndex;
      anchor.dataset.anchorType = anchorType;
      
      anchor.addEventListener('mousedown', function(e) {
        e.stopPropagation();
        e.preventDefault();
        startAnchorDrag(rectIndex, anchorType, e.clientX, e.clientY);
      });
      
      rectElement.appendChild(anchor);
    });
  }
  
  /**
   * Начинает drag для якоря (перемещение или изменение размера)
   */
  function startAnchorDrag(rectIndex, anchorType, clientX, clientY) {
    
    const rect = currentState.rectangles[rectIndex];
    if (!rect) return;
    
    const imgElement = document.getElementById('photoCardImg');
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgElement || !imgWrap) return;
    
    const imgRect = imgElement.getBoundingClientRect();
    const wrapRect = imgWrap.getBoundingClientRect();
    
    const relX = clientX - wrapRect.left;
    const relY = clientY - wrapRect.top;
    
    // Извлекаем координаты bbox используя ту же логику, что и в drawRectangles
    let bboxX, bboxY, bboxW, bboxH;
    if (rect.bbox && typeof rect.bbox === 'object') {
      // Формат: {bbox: {x, y, w, h}}
      bboxX = rect.bbox.x || 0;
      bboxY = rect.bbox.y || 0;
      bboxW = rect.bbox.w || 0;
      bboxH = rect.bbox.h || 0;
    } else {
      // Формат: {bbox_x, bbox_y, bbox_w, bbox_h}
      bboxX = rect.bbox_x || 0;
      bboxY = rect.bbox_y || 0;
      bboxW = rect.bbox_w || 0;
      bboxH = rect.bbox_h || 0;
    }
    
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    
    // Масштаб от натурального размера к отображаемому
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;
    
    // Смещение изображения внутри wrap (используем тот же способ, что и при отрисовке)
    const imgOffsetX_old = imgElement.offsetLeft || 0;
    const imgOffsetY_old = imgElement.offsetTop || 0;
    const imgOffsetX_new = imgRect.left - wrapRect.left;
    const imgOffsetY_new = imgRect.top - wrapRect.top;
    
    // Конвертируем в координаты относительно wrap (для отображения) - используем старый способ
    const originalX_old = bboxX * scaleX + imgOffsetX_old;
    const originalY_old = bboxY * scaleY + imgOffsetY_old;
    const originalW_old = bboxW * scaleX;
    const originalH_old = bboxH * scaleY;
    
    // Конвертируем используя функцию
    const displayBbox = convertBboxOriginalToDisplay(
      { x: bboxX, y: bboxY, w: bboxW, h: bboxH },
      imgElement,
      imgWrap
    );
    const originalX_new = displayBbox.x;
    const originalY_new = displayBbox.y;
    const originalW_new = displayBbox.w;
    const originalH_new = displayBbox.h;
    
    // Используем новые координаты (из функции)
    currentState.dragState = {
      rectIndex: rectIndex,
      anchorType: anchorType,
      startX: relX,
      startY: relY,
      originalX: originalX_new,
      originalY: originalY_new,
      originalW: originalW_new,
      originalH: originalH_new,
      scaleX: scaleX,
      scaleY: scaleY,
      imgOffsetX: imgOffsetX_new,
      imgOffsetY: imgOffsetY_new
    };
    
    document.addEventListener('mousemove', handleAnchorDrag);
    document.addEventListener('mouseup', endAnchorDrag);
  }
  
  /**
   * Обрабатывает перемещение якоря
   */
  function handleAnchorDrag(e) {
    if (!currentState.dragState) return;
    
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    const wrapRect = imgWrap.getBoundingClientRect();
    const relX = e.clientX - wrapRect.left;
    const relY = e.clientY - wrapRect.top;
    
    const dragState = currentState.dragState;
    const deltaX = relX - dragState.startX;
    const deltaY = relY - dragState.startY;
    
    let newX = dragState.originalX;
    let newY = dragState.originalY;
    let newW = dragState.originalW;
    let newH = dragState.originalH;
    
    const anchorType = dragState.anchorType;
    
    // Обрабатываем разные типы якорей
    if (anchorType === 'move') {
      // Перемещение всего rectangle
      newX = dragState.originalX + deltaX;
      newY = dragState.originalY + deltaY;
    } else if (anchorType === 'nw') {
      // Северо-западный угол: изменяем x, y, w, h
      newX = dragState.originalX + deltaX;
      newY = dragState.originalY + deltaY;
      newW = dragState.originalW - deltaX;
      newH = dragState.originalH - deltaY;
    } else if (anchorType === 'n') {
      // Северная сторона: изменяем y, h
      newY = dragState.originalY + deltaY;
      newH = dragState.originalH - deltaY;
    } else if (anchorType === 'ne') {
      // Северо-восточный угол: изменяем y, w, h
      newY = dragState.originalY + deltaY;
      newW = dragState.originalW + deltaX;
      newH = dragState.originalH - deltaY;
    } else if (anchorType === 'e') {
      // Восточная сторона: изменяем w
      newW = dragState.originalW + deltaX;
    } else if (anchorType === 'se') {
      // Юго-восточный угол: изменяем w, h
      newW = dragState.originalW + deltaX;
      newH = dragState.originalH + deltaY;
    } else if (anchorType === 's') {
      // Южная сторона: изменяем h
      newH = dragState.originalH + deltaY;
    } else if (anchorType === 'sw') {
      // Юго-западный угол: изменяем x, w, h
      newX = dragState.originalX + deltaX;
      newW = dragState.originalW - deltaX;
      newH = dragState.originalH + deltaY;
    } else if (anchorType === 'w') {
      // Западная сторона: изменяем x, w
      newX = dragState.originalX + deltaX;
      newW = dragState.originalW - deltaX;
    }
    
    // Получаем границы изображения для ограничения
    const imgElement = document.getElementById('photoCardImg');
    if (!imgElement) return;
    
    const imgRect = imgElement.getBoundingClientRect();
    // wrapRect уже объявлен выше, используем его
    const imgOffsetX = imgRect.left - wrapRect.left;
    const imgOffsetY = imgRect.top - wrapRect.top;
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    
    // Ограничиваем минимальный размер
    const minSize = 10;
    if (newW < minSize) {
      if (anchorType === 'nw' || anchorType === 'w' || anchorType === 'sw') {
        newX = dragState.originalX + dragState.originalW - minSize;
      }
      newW = minSize;
    }
    if (newH < minSize) {
      if (anchorType === 'nw' || anchorType === 'n' || anchorType === 'ne') {
        newY = dragState.originalY + dragState.originalH - minSize;
      }
      newH = minSize;
    }
    
    // Ограничиваем границами изображения
    // X координата не должна быть меньше смещения изображения
    if (newX < imgOffsetX) {
      const diff = imgOffsetX - newX;
      newX = imgOffsetX;
      if (anchorType === 'nw' || anchorType === 'w' || anchorType === 'sw') {
        newW = Math.max(minSize, newW - diff);
      }
    }
    // X + W не должна превышать правую границу изображения
    if (newX + newW > imgOffsetX + imgDisplayWidth) {
      newW = Math.max(minSize, imgOffsetX + imgDisplayWidth - newX);
    }
    // Y координата не должна быть меньше смещения изображения
    if (newY < imgOffsetY) {
      const diff = imgOffsetY - newY;
      newY = imgOffsetY;
      if (anchorType === 'nw' || anchorType === 'n' || anchorType === 'ne') {
        newH = Math.max(minSize, newH - diff);
      }
    }
    // Y + H не должна превышать нижнюю границу изображения
    if (newY + newH > imgOffsetY + imgDisplayHeight) {
      newH = Math.max(minSize, imgOffsetY + imgDisplayHeight - newY);
    }
    
    // Обновляем позицию и размер rectangle на экране
    const rectElement = imgWrap.querySelector(`[data-rect-index="${dragState.rectIndex}"]`);
    if (rectElement) {
      rectElement.style.left = `${newX}px`;
      rectElement.style.top = `${newY}px`;
      rectElement.style.width = `${newW}px`;
      rectElement.style.height = `${newH}px`;
    }
  }
  
  /**
   * Завершает drag якоря и сохраняет изменения
   */
  async function endAnchorDrag(e) {
    if (!currentState.dragState) return;
    
    document.removeEventListener('mousemove', handleAnchorDrag);
    document.removeEventListener('mouseup', endAnchorDrag);
    
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
    // Используем ту же логику, что и при отрисовке прямоугольников
    const imgNaturalWidth = imgElement.naturalWidth;
    const imgNaturalHeight = imgElement.naturalHeight;
    const imgRect = imgElement.getBoundingClientRect();
    const imgDisplayWidth = imgRect.width;
    const imgDisplayHeight = imgRect.height;
    
    // Масштаб от натурального размера к отображаемому
    const scaleX = imgDisplayWidth / imgNaturalWidth;
    const scaleY = imgDisplayHeight / imgNaturalHeight;
    
    const imgRect_end = imgElement.getBoundingClientRect();
    const wrapRect_end = imgWrap.getBoundingClientRect();
    
    // Получаем смещение изображения внутри wrap (используем тот же способ, что и при отрисовке)
    const imgOffsetX_old = imgElement.offsetLeft || 0;
    const imgOffsetY_old = imgElement.offsetTop || 0;
    const imgOffsetX_new = imgRect_end.left - wrapRect_end.left;
    const imgOffsetY_new = imgRect_end.top - wrapRect_end.top;
    
    // Получаем текущие координаты из стилей (они относительно imgWrap)
    const displayX = parseFloat(rectElement.style.left) || 0;
    const displayY = parseFloat(rectElement.style.top) || 0;
    const displayW = parseFloat(rectElement.style.width) || 0;
    const displayH = parseFloat(rectElement.style.height) || 0;
    
    // Используем функцию конвертации координат
    const originalBbox = convertBboxDisplayToOriginal(
      { x: displayX, y: displayY, w: displayW, h: displayH },
      imgElement,
      imgWrap
    );
    const naturalX = originalBbox.x;
    const naturalY = originalBbox.y;
    const naturalW = originalBbox.w;
    const naturalH = originalBbox.h;
    
    const rect = currentState.rectangles[dragState.rectIndex];
    if (!rect || !rect.id) {
      currentState.dragState = null;
      return;
    }
    
    // Сохраняем старое состояние для UNDO
    const oldBbox = rect.bbox || {};
    
    // Обновляем через API
    try {
      await updateRectangle(rect.id, {
        x: Math.round(naturalX),
        y: Math.round(naturalY),
        w: Math.round(naturalW),
        h: Math.round(naturalH)
      }, null, null);
      
      // Сохраняем действие в стек UNDO
      pushUndoAction({
        type: 'move_resize',
        rectangle_id: rect.id,
        oldBbox: oldBbox,
        newBbox: { x: naturalX, y: naturalY, w: naturalW, h: naturalH }
      });
      
      updateUndoButton();
      
      // Перезагружаем rectangles
      await loadRectangles();
    } catch (error) {
      console.error('[photo_card] Error updating rectangle:', error);
      alert('Ошибка при обновлении rectangle: ' + error.message);
      // Восстанавливаем исходное состояние
      drawRectangles();
    }
    
    currentState.dragState = null;
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
    // Если кликнули на уже выделенный rectangle - снимаем выделение
    if (currentState.selectedRectangleIndex === index) {
      currentState.selectedRectangleIndex = null;
    } else {
      currentState.selectedRectangleIndex = index;
    }
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
          rectangle_id: rect.id,
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
    // Преобразуем rectangle_id в число, если это возможно
    const rectangleIdInt = faceRectangleId !== null && faceRectangleId !== undefined 
      ? parseInt(faceRectangleId, 10) 
      : null;
    
    if (rectangleIdInt === null || isNaN(rectangleIdInt)) {
      throw new Error('rectangle_id is required and must be a valid number');
    }
    
    // Формируем payload в зависимости от режима
    const payload = {
      rectangle_id: rectangleIdInt
    };
    
    // Для сортируемых фото требуется pipeline_run_id
    if (currentState.mode === 'sorting') {
      if (!currentState.pipeline_run_id) {
        throw new Error('pipeline_run_id is required for sorting mode');
      }
      payload.pipeline_run_id = currentState.pipeline_run_id;
    } else {
      // Для архивных фото используем file_id или file_path
      if (currentState.file_id) {
        payload.file_id = currentState.file_id;
      } else if (currentState.file_path) {
        payload.path = currentState.file_path;
      } else {
        throw new Error('file_id or file_path is required for archive mode');
      }
    }
    
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
    
    // Сохраняем старое состояние для UNDO
    const rect = currentState.rectangles.find(r => r.id === faceRectangleId);
    if (rect) {
      pushUndoAction({
        type: 'delete_rectangle',
        rectangle_id: faceRectangleId,
        oldBbox: rect.bbox || {},
        oldPersonId: rect.person_id || null,
        oldAssignmentType: null
      });
    }
    
    // Формируем payload в зависимости от режима
    const payload = {
      rectangle_id: faceRectangleId
    };
    
    // Для сортируемых фото требуется pipeline_run_id
    if (currentState.mode === 'sorting') {
      if (!currentState.pipeline_run_id) {
        throw new Error('pipeline_run_id is required for sorting mode');
      }
      payload.pipeline_run_id = currentState.pipeline_run_id;
    } else {
      // Для архивных фото используем file_id или file_path
      if (currentState.file_id) {
        payload.file_id = currentState.file_id;
      } else if (currentState.file_path) {
        payload.path = currentState.file_path;
      } else {
        throw new Error('file_id or file_path is required for archive mode');
      }
    }
    
    const response = await fetch('/api/faces/rectangle/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
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
            duplicateMap[rect.rectangle_id] = true;
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
        rectangle_id: rect.id,
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
    
    // Копирование пути в буфер обмена
    const copyPathBtn = document.getElementById('photoCardCopyPath');
    if (copyPathBtn) {
      copyPathBtn.addEventListener('click', async function() {
        const path = currentState.file_path;
        if (!path) {
          alert('Путь к файлу не доступен');
          return;
        }
        
        try {
          await navigator.clipboard.writeText(path);
          // Временно меняем иконку на галочку для обратной связи
          const originalHTML = copyPathBtn.innerHTML;
          copyPathBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: block !important; width: 14px !important; height: 14px !important;"><path d="M13 4L6 11L3 8" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
          setTimeout(() => {
            copyPathBtn.innerHTML = originalHTML;
          }, 1000);
        } catch (error) {
          console.error('[photo_card] Failed to copy path:', error);
          // Fallback для старых браузеров
          try {
            const textArea = document.createElement('textarea');
            textArea.value = path;
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            document.body.appendChild(textArea);
            textArea.select();
            document.execCommand('copy');
            document.body.removeChild(textArea);
            
            const originalHTML = copyPathBtn.innerHTML;
            copyPathBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="display: block !important; width: 14px !important; height: 14px !important;"><path d="M13 4L6 11L3 8" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
            setTimeout(() => {
              copyPathBtn.innerHTML = originalHTML;
            }, 1000);
          } catch (fallbackError) {
            alert('Не удалось скопировать путь: ' + error.message);
          }
        }
      });
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
        try {
          const payload = {
            file_id: currentState.file_id,
            path: currentState.file_path
          };
          // Для архивных файлов pipeline_run_id не нужен
          if (currentState.pipeline_run_id) {
            payload.pipeline_run_id = currentState.pipeline_run_id;
          }
          
          const response = await fetch('/api/faces/rectangles/assign-outsider', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          if (response.ok) {
            await loadRectangles();
            await checkDuplicates();
          } else {
            const errorData = await response.json().catch(() => ({ detail: response.statusText }));
            alert('Ошибка: ' + (errorData.detail || response.statusText));
          }
        } catch (error) {
          console.error('[photo_card] Error assigning outsider:', error);
          alert('Ошибка: ' + error.message);
        }
      });
    }
    
    if (markAsCatBtn) {
      markAsCatBtn.addEventListener('click', async function() {
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
          }
        } catch (error) {
          console.error('[photo_card] Error marking as cat:', error);
          alert('Ошибка: ' + error.message);
        }
      });
    }
    
    if (markAsNoPeopleBtn) {
      markAsNoPeopleBtn.addEventListener('click', async function() {
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
          }
        } catch (error) {
          console.error('[photo_card] Error marking as no people:', error);
          alert('Ошибка: ' + error.message);
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
    
    // Кнопка "Привязать персону" с выпадающим меню
    const assignPersonBtn = document.getElementById('photoCardAssignPerson');
    const assignPersonContainer = document.getElementById('photoCardAssignPersonContainer');
    if (assignPersonBtn && assignPersonContainer) {
      assignPersonBtn.addEventListener('click', function(e) {
        if (currentState.isDrawing) {
          toggleDrawingMode();
        } else {
          // Проверяем, открыто ли меню
          const existingMenu = assignPersonContainer.querySelector('.assign-person-menu');
          if (existingMenu) {
            // Если меню открыто - закрываем его
            existingMenu.remove();
            e.stopPropagation();
          } else {
            // Если меню закрыто - открываем его
            showAssignPersonMenu(assignPersonContainer);
            e.stopPropagation();
          }
        }
      });
    }
    
    // Инициализация обработчиков рисования
    initDrawingHandlers();
  }
  
  /**
   * Инициализирует обработчики рисования
   */
  function initDrawingHandlers() {
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    // Удаляем старые обработчики, если есть
    detachDrawingHandlers();
    
    // Добавляем обработчик mousedown на imgWrap для начала рисования
    imgWrap.addEventListener('mousedown', handleDrawingStart);
  }
  
  /**
   * Прикрепляет обработчики рисования (алиас для initDrawingHandlers)
   */
  function attachDrawingHandlers() {
    initDrawingHandlers();
  }
  
  /**
   * Удаляет обработчики рисования
   */
  function detachDrawingHandlers() {
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    // Удаляем обработчик mousedown
    imgWrap.removeEventListener('mousedown', handleDrawingStart);
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
          await updateRectangle(action.rectangle_id, action.oldBbox, action.oldPersonId, action.oldAssignmentType);
          await loadRectangles();
          await checkDuplicates();
          break;
        case 'delete_rectangle':
          // Восстанавливаем rectangle (нужен API для восстановления)
          alert('Восстановление удаленного rectangle пока не реализовано');
          break;
        case 'assign_person':
          // Удаляем привязку персоны
          await updateRectangle(action.rectangle_id, null, null, null);
          await loadRectangles();
          await checkDuplicates();
          break;
        case 'change_assignment_type':
          // Восстанавливаем предыдущий тип привязки
          await updateRectangle(action.rectangle_id, null, action.person_id, action.oldAssignmentType);
          await loadRectangles();
          await checkDuplicates();
          break;
        case 'move_resize':
          // Восстанавливаем предыдущие координаты и размеры
          await updateRectangle(action.rectangle_id, action.oldBbox, null, null);
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

  /**
   * Показывает меню "Привязать персону"
   */
  function showAssignPersonMenu(container) {
    // Закрываем все открытые меню
    document.querySelectorAll('.assign-person-menu').forEach(menu => {
      menu.remove();
    });
    
    const menu = document.createElement('div');
    menu.className = 'assign-person-menu';
    menu.style.position = 'absolute';
    menu.style.left = '0';
    menu.style.bottom = '100%';
    menu.style.marginBottom = '4px';
    menu.style.marginTop = '4px';
    menu.style.background = '#fff';
    menu.style.border = '1px solid #e5e7eb';
    menu.style.borderRadius = '8px';
    menu.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    menu.style.minWidth = '200px';
    menu.style.zIndex = '10001';
    menu.style.padding = '4px 0';
    
    // Проверяем, есть ли rectangles на фото
    const hasRectangles = currentState.rectangles && currentState.rectangles.length > 0;
    
    // Пункт "указать область на фото"
    const areaContainer = document.createElement('div');
    areaContainer.style.position = 'relative';
    
    const areaBtn = document.createElement('button');
    areaBtn.textContent = 'указать область на фото';
    areaBtn.style.width = '100%';
    areaBtn.style.padding = '10px 12px';
    areaBtn.style.textAlign = 'left';
    areaBtn.style.border = 'none';
    areaBtn.style.background = '#fff';
    areaBtn.style.cursor = 'pointer';
    areaBtn.style.fontSize = '14px';
    areaBtn.style.display = 'flex';
    areaBtn.style.justifyContent = 'space-between';
    areaBtn.style.alignItems = 'center';
    areaBtn.innerHTML = '<span>указать область на фото</span><span style="margin-left: 8px;">▶</span>';
    
    areaBtn.addEventListener('mouseenter', function(e) {
      e.stopPropagation();
      showAreaSubmenu(areaContainer);
    });
    
    areaContainer.appendChild(areaBtn);
    menu.appendChild(areaContainer);
    
    // Пункт "к фото целиком"
    const wholePhotoBtn = document.createElement('button');
    wholePhotoBtn.textContent = 'к фото целиком';
    wholePhotoBtn.style.width = '100%';
    wholePhotoBtn.style.padding = '10px 12px';
    wholePhotoBtn.style.textAlign = 'left';
    wholePhotoBtn.style.border = 'none';
    wholePhotoBtn.style.background = '#fff';
    wholePhotoBtn.style.cursor = hasRectangles ? 'not-allowed' : 'pointer';
    wholePhotoBtn.style.fontSize = '14px';
    wholePhotoBtn.style.opacity = hasRectangles ? '0.5' : '1';
    wholePhotoBtn.disabled = hasRectangles;
    
    if (!hasRectangles) {
      wholePhotoBtn.addEventListener('click', async function(e) {
        e.stopPropagation();
        menu.remove();
        await showPersonDialogForWholePhoto();
      });
    }
    
    menu.appendChild(wholePhotoBtn);
    
    // Закрытие меню при клике вне его
    const closeMenu = (e) => {
      if (!menu.contains(e.target) && !container.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };
    
    setTimeout(() => {
      document.addEventListener('click', closeMenu);
    }, 100);
    
    container.appendChild(menu);
  }
  
  /**
   * Показывает подменю для "указать область на фото"
   */
  function showAreaSubmenu(container) {
    // Закрываем все открытые подменю
    document.querySelectorAll('.area-submenu').forEach(submenu => {
      submenu.remove();
    });
    
    const submenu = document.createElement('div');
    submenu.className = 'area-submenu';
    submenu.style.position = 'absolute';
    submenu.style.left = '100%';
    submenu.style.top = '0';
    submenu.style.marginLeft = '4px';
    submenu.style.background = '#fff';
    submenu.style.border = '1px solid #e5e7eb';
    submenu.style.borderRadius = '8px';
    submenu.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    submenu.style.minWidth = '150px';
    submenu.style.zIndex = '10002';
    submenu.style.padding = '4px 0';
    
    // Пункт "лицо"
    const faceBtn = document.createElement('button');
    faceBtn.textContent = 'лицо';
    faceBtn.style.width = '100%';
    faceBtn.style.padding = '10px 12px';
    faceBtn.style.textAlign = 'left';
    faceBtn.style.border = 'none';
    faceBtn.style.background = '#fff';
    faceBtn.style.cursor = 'pointer';
    faceBtn.style.fontSize = '14px';
    
    faceBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      submenu.remove();
      document.querySelectorAll('.assign-person-menu').forEach(menu => {
        menu.remove();
      });
      startDrawingMode(true); // true = лицо
    });
    
    submenu.appendChild(faceBtn);
    
    // Пункт "без лица"
    const noFaceBtn = document.createElement('button');
    noFaceBtn.textContent = 'без лица';
    noFaceBtn.style.width = '100%';
    noFaceBtn.style.padding = '10px 12px';
    noFaceBtn.style.textAlign = 'left';
    noFaceBtn.style.border = 'none';
    noFaceBtn.style.background = '#fff';
    noFaceBtn.style.cursor = 'pointer';
    noFaceBtn.style.fontSize = '14px';
    
    noFaceBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      submenu.remove();
      document.querySelectorAll('.assign-person-menu').forEach(menu => {
        menu.remove();
      });
      startDrawingMode(false); // false = без лица
    });
    
    submenu.appendChild(noFaceBtn);
    
    container.appendChild(submenu);
  }
  
  /**
   * Запускает режим рисования
   */
  function startDrawingMode(isFace) {
    currentState.isDrawing = true;
    currentState.drawingIsFace = isFace; // Сохраняем тип (лицо или без лица)
    
    const assignPersonBtn = document.getElementById('photoCardAssignPerson');
    if (assignPersonBtn) {
      assignPersonBtn.textContent = 'Отменить рисование';
      assignPersonBtn.style.background = '#fef3c7';
    }
    
    // Выходим из режима редактирования при включении рисования
    if (currentState.isEditMode) {
      toggleEditMode();
    }
  }
  
  /**
   * Переключает режим рисования (для отмены)
   */
  function toggleDrawingMode() {
    currentState.isDrawing = !currentState.isDrawing;
    const assignPersonBtn = document.getElementById('photoCardAssignPerson');
    if (assignPersonBtn) {
      assignPersonBtn.textContent = 'Привязать персону ▶';
      assignPersonBtn.style.background = '';
    }
    
    // Очищаем временный rectangle при выходе из режима рисования
    if (!currentState.isDrawing && currentState.drawingState) {
      cleanupDrawing();
    }
  }
  
  /**
   * Показывает диалог выбора типа rectangle (лицо или персона)
   */
  function showRectangleTypeDialog(bbox) {
    // Используем сохраненный тип из drawingIsFace
    const isFace = currentState.drawingIsFace !== undefined ? currentState.drawingIsFace : true;
    
    if (isFace) {
      // Это лицо - открываем диалог выбора персоны с переключателем
      showPersonDialogForNewRectangle(bbox, true);
    } else {
      // Без лица - открываем диалог выбора персоны с переключателем (но переключатель будет выключен)
      showPersonDialogForNewRectangle(bbox, false);
    }
  }
  
  /**
   * Обработчик начала рисования
   */
  function handleDrawingStart(e) {
    if (!currentState.isDrawing) return;
    
    // Игнорируем клики на существующие rectangles
    if (e.target.classList.contains('photo-card-rectangle') || 
        e.target.closest('.photo-card-rectangle') ||
        e.target.classList.contains('photo-card-rectangle-anchor') ||
        e.target.closest('.photo-card-rectangle-anchor')) {
      return;
    }
    
    // Предотвращаем стандартное поведение (drag изображения)
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
    
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    const rect = imgWrap.getBoundingClientRect();
    const startX = e.clientX - rect.left;
    const startY = e.clientY - rect.top;
    
    currentState.drawingState = {
      startX: startX,
      startY: startY,
      currentX: startX,
      currentY: startY
    };
    
    // Создаем временный rectangle элемент
    createTempDrawingRect();
    
    document.addEventListener('mousemove', handleDrawingMove);
    document.addEventListener('mouseup', handleDrawingEnd);
  }
  
  /**
   * Обработчик движения мыши при рисовании
   */
  function handleDrawingMove(e) {
    if (!currentState.isDrawing || !currentState.drawingState) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    const rect = imgWrap.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    currentState.drawingState.currentX = currentX;
    currentState.drawingState.currentY = currentY;
    
    updateTempDrawingRect();
  }
  
  /**
   * Обработчик завершения рисования
   */
  function handleDrawingEnd(e) {
    if (!currentState.isDrawing || !currentState.drawingState) return;
    
    e.preventDefault();
    e.stopPropagation();
    
    document.removeEventListener('mousemove', handleDrawingMove);
    document.removeEventListener('mouseup', handleDrawingEnd);
    
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap) return;
    
    const rect = imgWrap.getBoundingClientRect();
    const endX = e.clientX - rect.left;
    const endY = e.clientY - rect.top;
    
    const startX = currentState.drawingState.startX;
    const startY = currentState.drawingState.startY;
    
    // Вычисляем размеры rectangle
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const w = Math.abs(endX - startX);
    const h = Math.abs(endY - startY);
    
    // Минимальный размер rectangle
    if (w < 10 || h < 10) {
      cleanupDrawing();
      return;
    }
    
    // Конвертируем координаты в координаты изображения
    const imgElement = document.getElementById('photoCardImg');
    if (!imgElement) {
      cleanupDrawing();
      return;
    }
    
    // Используем функцию конвертации координат
    const bbox = convertBboxDisplayToOriginal(
      { x, y, w, h },
      imgElement,
      imgWrap
    );
    
    // Очищаем временный rectangle
    cleanupDrawing();
    
    // Показываем диалог выбора типа
    showRectangleTypeDialog(bbox);
  }
  
  /**
   * Создает временный rectangle для визуализации рисования
   */
  function createTempDrawingRect() {
    const imgWrap = document.getElementById('photoCardImgWrap');
    if (!imgWrap || !currentState.drawingState) return;
    
    const tempRect = document.createElement('div');
    tempRect.className = 'photo-card-rectangle photo-card-rectangle-drawing';
    tempRect.style.position = 'absolute';
    tempRect.style.border = '2px dashed #3b82f6';
    tempRect.style.backgroundColor = 'rgba(59, 130, 246, 0.1)';
    tempRect.style.pointerEvents = 'none';
    tempRect.style.zIndex = '10000';
    
    currentState.drawingState.tempRectElement = tempRect;
    imgWrap.appendChild(tempRect);
    
    updateTempDrawingRect();
  }
  
  /**
   * Обновляет позицию и размер временного rectangle
   */
  function updateTempDrawingRect() {
    if (!currentState.drawingState || !currentState.drawingState.tempRectElement) return;
    
    const state = currentState.drawingState;
    const x = Math.min(state.startX, state.currentX);
    const y = Math.min(state.startY, state.currentY);
    const w = Math.abs(state.currentX - state.startX);
    const h = Math.abs(state.currentY - state.startY);
    
    const tempRect = currentState.drawingState.tempRectElement;
    tempRect.style.left = x + 'px';
    tempRect.style.top = y + 'px';
    tempRect.style.width = w + 'px';
    tempRect.style.height = h + 'px';
  }
  
  /**
   * Очищает временный rectangle
   */
  function cleanupDrawing() {
    if (currentState.drawingState && currentState.drawingState.tempRectElement) {
      currentState.drawingState.tempRectElement.remove();
    }
    currentState.drawingState = null;
  }
  
  /**
   * Показывает диалог выбора персоны для нового rectangle (лицо)
   */
  async function showPersonDialogForNewRectangle(bbox, isFace) {
    // Загружаем список персон, если еще не загружен
    if (!currentState.allPersons || currentState.allPersons.length === 0) {
      await loadPersons();
    }
    
    // Создаем модальное окно для выбора персоны
    const modal = document.createElement('div');
    modal.className = 'person-dialog-modal';
    modal.style.position = 'fixed';
    modal.style.inset = '0';
    modal.style.background = 'rgba(0,0,0,0.5)';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.zIndex = '10001';
    
    const dialog = document.createElement('div');
    dialog.style.background = '#fff';
    dialog.style.borderRadius = '12px';
    dialog.style.padding = '24px';
    dialog.style.minWidth = '400px';
    dialog.style.maxWidth = '600px';
    dialog.style.maxHeight = '80vh';
    dialog.style.overflow = 'auto';
    dialog.style.boxShadow = '0 20px 60px rgba(0,0,0,0.3)';
    
    // Заголовок
    const title = document.createElement('h3');
    title.textContent = 'Выберите персону';
    title.style.margin = '0 0 16px 0';
    title.style.fontSize = '18px';
    title.style.fontWeight = '600';
    dialog.appendChild(title);
    
    // Переключатель "лицо" / "без лица"
    const faceToggleContainer = document.createElement('div');
    faceToggleContainer.style.marginBottom = '16px';
    faceToggleContainer.style.display = 'flex';
    faceToggleContainer.style.alignItems = 'center';
    faceToggleContainer.style.gap = '12px';
    
    const faceToggleLabel = document.createElement('label');
    faceToggleLabel.style.display = 'flex';
    faceToggleLabel.style.alignItems = 'center';
    faceToggleLabel.style.gap = '8px';
    faceToggleLabel.style.cursor = 'pointer';
    
    const faceToggle = document.createElement('input');
    faceToggle.type = 'checkbox';
    faceToggle.checked = isFace; // По умолчанию "лицо"
    faceToggle.style.width = '18px';
    faceToggle.style.height = '18px';
    faceToggle.style.cursor = 'pointer';
    
    const faceToggleText = document.createElement('span');
    faceToggleText.textContent = 'Лицо';
    faceToggleText.style.fontSize = '14px';
    
    faceToggleLabel.appendChild(faceToggle);
    faceToggleLabel.appendChild(faceToggleText);
    faceToggleContainer.appendChild(faceToggleLabel);
    dialog.appendChild(faceToggleContainer);
    
    // Список персон (иерархический)
    const personsList = document.createElement('div');
    personsList.style.maxHeight = '400px';
    personsList.style.overflowY = 'auto';
    
    // Группируем персон по группам (исключая "Посторонний" по флагу is_ignored)
    const personsByGroup = {};
    const noGroupPersons = [];
    
    currentState.allPersons.forEach(person => {
      // Исключаем "Посторонний" по флагу is_ignored (будет добавлен отдельной кнопкой в конце)
      if (person.is_ignored === true) return;
      
      const group = person.group || null;
      if (group) {
        if (!personsByGroup[group]) {
          personsByGroup[group] = [];
        }
        personsByGroup[group].push(person);
      } else {
        noGroupPersons.push(person);
      }
    });
    
    // Добавляем группы
    const sortedGroups = Object.keys(personsByGroup).sort();
    sortedGroups.forEach(groupName => {
      const groupLabel = document.createElement('div');
      groupLabel.textContent = groupName;
      groupLabel.style.fontWeight = '600';
      groupLabel.style.fontSize = '12px';
      groupLabel.style.color = '#6b7280';
      groupLabel.style.marginTop = '16px';
      groupLabel.style.marginBottom = '8px';
      groupLabel.style.textTransform = 'uppercase';
      personsList.appendChild(groupLabel);
      
      personsByGroup[groupName].forEach(person => {
        const personBtn = document.createElement('button');
        personBtn.textContent = person.name;
        personBtn.style.width = '100%';
        personBtn.style.padding = '10px 12px';
        personBtn.style.textAlign = 'left';
        personBtn.style.border = '1px solid #e5e7eb';
        personBtn.style.borderRadius = '8px';
        personBtn.style.background = '#fff';
        personBtn.style.cursor = 'pointer';
        personBtn.style.marginBottom = '4px';
        personBtn.style.fontSize = '14px';
        personBtn.style.transition = 'background 0.15s';
        
        personBtn.addEventListener('mouseenter', () => {
          personBtn.style.background = '#f9fafb';
        });
        personBtn.addEventListener('mouseleave', () => {
          personBtn.style.background = '#fff';
        });
        
        personBtn.addEventListener('click', () => {
          modal.remove();
          const isFaceValue = faceToggle.checked;
          createRectangle(bbox, person.id, isFaceValue);
        });
        
        personsList.appendChild(personBtn);
      });
    });
    
    // Добавляем персон без группы
    noGroupPersons.forEach(person => {
      const personBtn = document.createElement('button');
      personBtn.textContent = person.name;
      personBtn.style.width = '100%';
      personBtn.style.padding = '10px 12px';
      personBtn.style.textAlign = 'left';
      personBtn.style.border = '1px solid #e5e7eb';
      personBtn.style.borderRadius = '8px';
      personBtn.style.background = '#fff';
      personBtn.style.cursor = 'pointer';
      personBtn.style.marginBottom = '4px';
      personBtn.style.fontSize = '14px';
      personBtn.style.transition = 'background 0.15s';
      
      personBtn.addEventListener('mouseenter', () => {
        personBtn.style.background = '#f9fafb';
      });
      personBtn.addEventListener('mouseleave', () => {
        personBtn.style.background = '#fff';
      });
      
      personBtn.addEventListener('click', () => {
        modal.remove();
        const isFaceValue = faceToggle.checked;
        createRectangle(bbox, person.id, isFaceValue);
      });
      
      personsList.appendChild(personBtn);
    });
    
    // Кнопка "Посторонние"
    const outsiderBtn = document.createElement('button');
    outsiderBtn.textContent = 'Посторонний';
    outsiderBtn.style.width = '100%';
    outsiderBtn.style.padding = '10px 12px';
    outsiderBtn.style.textAlign = 'left';
    outsiderBtn.style.border = '1px solid #e5e7eb';
    outsiderBtn.style.borderRadius = '8px';
    outsiderBtn.style.background = '#fff';
    outsiderBtn.style.cursor = 'pointer';
    outsiderBtn.style.marginTop = '16px';
    outsiderBtn.style.fontSize = '14px';
    outsiderBtn.style.transition = 'background 0.15s';
    
    outsiderBtn.addEventListener('mouseenter', () => {
      outsiderBtn.style.background = '#f9fafb';
    });
    outsiderBtn.addEventListener('mouseleave', () => {
      outsiderBtn.style.background = '#fff';
    });
    
    outsiderBtn.addEventListener('click', async () => {
      const outsiderPerson = currentState.allPersons.find(p => p.is_ignored === true);
      if (!outsiderPerson) {
        alert('Персона "Посторонний" не найдена');
        return;
      }
      modal.remove();
      const isFaceValue = faceToggle.checked;
      createRectangle(bbox, outsiderPerson.id, isFaceValue);
    });
    
    personsList.appendChild(outsiderBtn);
    
    // Кнопка "Отмена"
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Отмена';
    cancelBtn.style.width = '100%';
    cancelBtn.style.padding = '10px 12px';
    cancelBtn.style.marginTop = '16px';
    cancelBtn.style.border = '1px solid #d1d5db';
    cancelBtn.style.borderRadius = '8px';
    cancelBtn.style.background = '#fff';
    cancelBtn.style.cursor = 'pointer';
    cancelBtn.style.fontSize = '14px';
    
    cancelBtn.addEventListener('click', () => {
      modal.remove();
    });
    
    dialog.appendChild(personsList);
    dialog.appendChild(cancelBtn);
    modal.appendChild(dialog);
    
    // Закрытие по клику вне диалога
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.remove();
      }
    });
    
    document.body.appendChild(modal);
  }
  
  /**
   * Создает новый rectangle через API
   */
  async function createRectangle(bbox, personId, isFace) {
    try {
      // Формируем payload в зависимости от режима
      const payload = {
        bbox: bbox
      };
      
      // Для сортируемых фото требуется pipeline_run_id И file_id/path
      if (currentState.mode === 'sorting') {
        if (!currentState.pipeline_run_id) {
          throw new Error('pipeline_run_id is required for sorting mode');
        }
        payload.pipeline_run_id = currentState.pipeline_run_id;
        // Для сортируемых фото тоже нужен file_id или path
        if (currentState.file_id) {
          payload.file_id = currentState.file_id;
        } else if (currentState.file_path) {
          payload.path = currentState.file_path;
        } else {
          throw new Error('file_id or file_path is required for sorting mode');
        }
      } else {
        // Для архивных фото используем file_id или file_path
        if (currentState.file_id) {
          payload.file_id = currentState.file_id;
        } else if (currentState.file_path) {
          payload.path = currentState.file_path;
        } else {
          throw new Error('file_id or file_path is required for archive mode');
        }
      }
      
      // Если персона назначена, добавляем person_id и assignment_type
      if (personId !== null && personId !== undefined) {
        payload.person_id = personId;
        payload.assignment_type = 'manual_face';
      }
      
      const response = await fetch('/api/faces/rectangle/create', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to create rectangle');
      }
      
      // Перезагружаем rectangles
      await loadRectangles();
      await checkDuplicates();
      
      // Выходим из режима рисования
      if (currentState.isDrawing) {
        currentState.isDrawing = false;
        currentState.drawingIsFace = undefined;
        const assignPersonBtn = document.getElementById('photoCardAssignPerson');
        if (assignPersonBtn) {
          assignPersonBtn.textContent = 'Привязать персону ▶';
          assignPersonBtn.style.background = '';
        }
      }
    } catch (error) {
      console.error('[photo_card] Error creating rectangle:', error);
      alert('Ошибка при создании rectangle: ' + error.message);
    }
  }
  
  /**
   * Показывает диалог выбора персоны для всего фото
   */
  async function showPersonDialogForWholePhoto() {
    // Загружаем список персон, если еще не загружен
    if (!currentState.allPersons || currentState.allPersons.length === 0) {
      await loadPersons();
    }
    
    // Создаем модальное окно для выбора персоны
    const modal = document.createElement('div');
    modal.className = 'person-dialog-modal';
    modal.style.position = 'fixed';
    modal.style.inset = '0';
    modal.style.background = 'rgba(0,0,0,0.5)';
    modal.style.display = 'flex';
    modal.style.alignItems = 'center';
    modal.style.justifyContent = 'center';
    modal.style.zIndex = '10001';
    
    const dialog = document.createElement('div');
    dialog.style.background = '#fff';
    dialog.style.borderRadius = '12px';
    dialog.style.padding = '24px';
    dialog.style.minWidth = '400px';
    dialog.style.maxWidth = '600px';
    dialog.style.maxHeight = '80vh';
    dialog.style.overflow = 'auto';
    dialog.style.boxShadow = '0 20px 60px rgba(0,0,0,0.3)';
    
    // Заголовок
    const title = document.createElement('h3');
    title.textContent = 'Выберите персону для фото';
    title.style.margin = '0 0 16px 0';
    title.style.fontSize = '18px';
    title.style.fontWeight = '600';
    dialog.appendChild(title);
    
    // Список персон (иерархический) - используем ту же логику, что и в showPersonDialogForNewRectangle
    const personsList = document.createElement('div');
    personsList.style.maxHeight = '400px';
    personsList.style.overflowY = 'auto';
    
    // Группируем персон по группам (исключая "Посторонний" по флагу is_ignored)
    const personsByGroup = {};
    const noGroupPersons = [];
    
    currentState.allPersons.forEach(person => {
      // Исключаем "Посторонний" по флагу is_ignored (будет добавлен отдельной кнопкой в конце)
      if (person.is_ignored === true) return;
      
      const group = person.group || null;
      if (group) {
        if (!personsByGroup[group]) {
          personsByGroup[group] = [];
        }
        personsByGroup[group].push(person);
      } else {
        noGroupPersons.push(person);
      }
    });
    
    // Добавляем группы
    const sortedGroups = Object.keys(personsByGroup).sort();
    sortedGroups.forEach(groupName => {
      const groupLabel = document.createElement('div');
      groupLabel.textContent = groupName;
      groupLabel.style.fontWeight = '600';
      groupLabel.style.fontSize = '12px';
      groupLabel.style.color = '#6b7280';
      groupLabel.style.marginTop = '16px';
      groupLabel.style.marginBottom = '8px';
      groupLabel.style.textTransform = 'uppercase';
      personsList.appendChild(groupLabel);
      
      personsByGroup[groupName].forEach(person => {
        const personBtn = document.createElement('button');
        personBtn.textContent = person.name;
        personBtn.style.width = '100%';
        personBtn.style.padding = '10px 12px';
        personBtn.style.textAlign = 'left';
        personBtn.style.border = '1px solid #e5e7eb';
        personBtn.style.borderRadius = '8px';
        personBtn.style.background = '#fff';
        personBtn.style.cursor = 'pointer';
        personBtn.style.marginBottom = '4px';
        personBtn.style.fontSize = '14px';
        personBtn.style.transition = 'background 0.15s';
        
        personBtn.addEventListener('mouseenter', () => {
          personBtn.style.background = '#f9fafb';
        });
        personBtn.addEventListener('mouseleave', () => {
          personBtn.style.background = '#fff';
        });
        
        personBtn.addEventListener('click', () => {
          modal.remove();
          assignPersonToWholePhoto(person.id);
        });
        
        personsList.appendChild(personBtn);
      });
    });
    
    // Добавляем персон без группы
    noGroupPersons.forEach(person => {
      const personBtn = document.createElement('button');
      personBtn.textContent = person.name;
      personBtn.style.width = '100%';
      personBtn.style.padding = '10px 12px';
      personBtn.style.textAlign = 'left';
      personBtn.style.border = '1px solid #e5e7eb';
      personBtn.style.borderRadius = '8px';
      personBtn.style.background = '#fff';
      personBtn.style.cursor = 'pointer';
      personBtn.style.marginBottom = '4px';
      personBtn.style.fontSize = '14px';
      personBtn.style.transition = 'background 0.15s';
      
      personBtn.addEventListener('mouseenter', () => {
        personBtn.style.background = '#f9fafb';
      });
      personBtn.addEventListener('mouseleave', () => {
        personBtn.style.background = '#fff';
      });
      
      personBtn.addEventListener('click', () => {
        modal.remove();
        assignPersonToWholePhoto(person.id);
      });
      
      personsList.appendChild(personBtn);
    });
    
    // Кнопка "Посторонние"
    const outsiderBtn = document.createElement('button');
    outsiderBtn.textContent = 'Посторонний';
    outsiderBtn.style.width = '100%';
    outsiderBtn.style.padding = '10px 12px';
    outsiderBtn.style.textAlign = 'left';
    outsiderBtn.style.border = '1px solid #e5e7eb';
    outsiderBtn.style.borderRadius = '8px';
    outsiderBtn.style.background = '#fff';
    outsiderBtn.style.cursor = 'pointer';
    outsiderBtn.style.marginTop = '16px';
    outsiderBtn.style.fontSize = '14px';
    outsiderBtn.style.transition = 'background 0.15s';
    
    outsiderBtn.addEventListener('mouseenter', () => {
      outsiderBtn.style.background = '#f9fafb';
    });
    outsiderBtn.addEventListener('mouseleave', () => {
      outsiderBtn.style.background = '#fff';
    });
    
    outsiderBtn.addEventListener('click', async () => {
      const outsiderPerson = currentState.allPersons.find(p => p.is_ignored === true);
      if (!outsiderPerson) {
        alert('Персона "Посторонний" не найдена');
        return;
      }
      modal.remove();
      assignPersonToWholePhoto(outsiderPerson.id);
    });
    
    personsList.appendChild(outsiderBtn);
    
    // Кнопка "Отмена"
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Отмена';
    cancelBtn.style.width = '100%';
    cancelBtn.style.padding = '10px 12px';
    cancelBtn.style.marginTop = '16px';
    cancelBtn.style.border = '1px solid #d1d5db';
    cancelBtn.style.borderRadius = '8px';
    cancelBtn.style.background = '#fff';
    cancelBtn.style.cursor = 'pointer';
    cancelBtn.style.fontSize = '14px';
    
    cancelBtn.addEventListener('click', () => {
      modal.remove();
    });
    
    dialog.appendChild(personsList);
    dialog.appendChild(cancelBtn);
    modal.appendChild(dialog);
    
    // Закрытие по клику вне диалога
    modal.addEventListener('click', (e) => {
      if (e.target === modal) {
        modal.remove();
      }
    });
    
    document.body.appendChild(modal);
  }
  
  /**
   * Привязывает персону ко всему фото (через file_persons)
   */
  async function assignPersonToWholePhoto(personId) {
    try {
      // Формируем payload в зависимости от режима
      const payload = {
        person_id: personId
      };
      
      // Для сортируемых фото требуется pipeline_run_id
      if (currentState.mode === 'sorting') {
        if (!currentState.pipeline_run_id) {
          throw new Error('pipeline_run_id is required for sorting mode');
        }
        payload.pipeline_run_id = currentState.pipeline_run_id;
      } else {
        // Для архивных фото используем file_id или file_path
        if (currentState.file_id) {
          payload.file_id = currentState.file_id;
        } else if (currentState.file_path) {
          payload.path = currentState.file_path;
        } else {
          throw new Error('file_id or file_path is required for archive mode');
        }
      }
      
      // Используем существующий endpoint, но нужно добавить поддержку архивных файлов
      const endpoint = currentState.mode === 'sorting' 
        ? '/api/persons/assign-file'
        : '/api/faces/file/assign-person'; // Новый endpoint для архивных файлов
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to assign person to file');
      }
      
      // Перезагружаем данные
      await loadRectangles();
      await checkDuplicates();
    } catch (error) {
      console.error('[photo_card] Error assigning person to whole photo:', error);
      alert('Ошибка при привязке персоны: ' + error.message);
    }
  }

  // Экспортируем функцию openPhotoCard в глобальную область
  window.openPhotoCard = openPhotoCard;
  window.closePhotoCard = closePhotoCard;

})();
