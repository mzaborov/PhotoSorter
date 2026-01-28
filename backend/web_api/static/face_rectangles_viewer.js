/**
 * Общий модуль для отображения фото с прямоугольниками вокруг лиц
 * Используется на странице персоны, странице кластера и других местах
 */

// Общая функция для формирования подписи лица
function formatFaceLabel(personName, isMe, clusterId) {
  let labelText = '';
  if (personName) {
    labelText = personName + (isMe ? ' (я)' : '');
    // Добавляем номер кластера, если он есть
    if (clusterId) {
      labelText += ` (Кластер #${clusterId})`;
    }
  } else {
    labelText = clusterId ? `Кластер #${clusterId}` : 'Не назначено';
  }
  return labelText;
}

/**
 * Рисует все лица на фото (желтые прямоугольники), кроме текущего
 * @param {Array} allFacesOnImage - Массив всех лиц на фото
 * @param {number|string} currentFaceId - ID текущего лица (будет пропущено)
 * @param {Object} originalImageSize - Размеры исходного изображения {width, height, exif_orientation}
 * @param {Function} onFaceClick - Callback при клике на лицо (faceId) -> void
 */
function drawAllFaceRectangles(allFacesOnImage, currentFaceId, originalImageSize, onFaceClick) {
  const lbImg = document.getElementById('lbImg');
  const lbBody = document.getElementById('lbBody');
  try {
    if (!lbImg || !lbBody || !lbImg.complete) {
      return;
    }
  } catch (e) {
    return;
  }

  // Удаляем старые желтые rectangles
  const oldRects = lbBody.querySelectorAll('.lb-rectangle.other, .lb-rectangle-label:not(.current)');
  oldRects.forEach(el => el.remove());

  if (!allFacesOnImage || allFacesOnImage.length === 0) {
    return;
  }

  // Получаем размеры изображения
  const imgRect = lbImg.getBoundingClientRect();
  const imgNaturalWidth = lbImg.naturalWidth;
  const imgNaturalHeight = lbImg.naturalHeight;
  const imgDisplayWidth = imgRect.width;
  const imgDisplayHeight = imgRect.height;

  if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
    return;
  }
  
  // Рисуем каждое лицо (кроме текущего)
  let drawnCount = 0;
  for (const otherFace of allFacesOnImage) {
    // Сравниваем как числа и как строки
    const otherFaceIdNum = Number(otherFace.face_id);
    const currentFaceIdNum = Number(currentFaceId);
    const isCurrent = otherFaceIdNum === currentFaceIdNum || 
                     otherFace.face_id === currentFaceId ||
                     String(otherFace.face_id) === String(currentFaceId);
    
    if (isCurrent) {
      continue; // Пропускаем текущее лицо (оно будет нарисовано синим в конце)
    }
    
    let bboxX = otherFace.bbox.x;
    let bboxY = otherFace.bbox.y;
    let bboxW = otherFace.bbox.w;
    let bboxH = otherFace.bbox.h;
    
    // Масштабируем координаты
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
        const scaleX = imgNaturalWidth / originalWidth;
        const scaleY = imgNaturalHeight / originalHeight;
        bboxX = bboxX * scaleX;
        bboxY = bboxY * scaleY;
        bboxW = bboxW * scaleX;
        bboxH = bboxH * scaleY;
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
    const rectX = bboxX * scaleX;
    const rectY = bboxY * scaleY;
    const rectW = bboxW * scaleX;
    const rectH = bboxH * scaleY;
    
    // Позиционируем относительно body
    const bodyRect = lbBody.getBoundingClientRect();
    const left = (imgRect.left - bodyRect.left) + rectX;
    const top = (imgRect.top - bodyRect.top) + rectY;
    
    // Создаем желтый rectangle
    const rect = document.createElement('div');
    rect.className = 'lb-rectangle other';
    rect.style.left = `${left}px`;
    rect.style.top = `${top}px`;
    rect.style.width = `${rectW}px`;
    rect.style.height = `${rectH}px`;
    rect.style.display = 'block';
    rect.style.position = 'absolute';
    rect.style.zIndex = '1000';
    rect.style.cursor = 'pointer';
    rect.style.pointerEvents = 'auto';
    
    // Добавляем клик для перехода к кластеру
    if (otherFace.cluster_id) {
      rect.onclick = function(e) {
        e.stopPropagation();
        const clusterUrl = `/face-clusters/${otherFace.cluster_id}`;
        window.open(clusterUrl, '_blank');
      };
      rect.title = (otherFace.person_name || '') + (otherFace.cluster_id ? ` | Кластер #${otherFace.cluster_id} (клик для открытия)` : '');
    } else if (onFaceClick) {
      rect.onclick = function(e) {
        e.stopPropagation();
        onFaceClick(otherFace.face_id);
      };
      rect.title = (otherFace.person_name || '') + ' (клик для назначения персоны)';
    }
    
    lbBody.appendChild(rect);
    
    // Добавляем подпись с именем персоны и номером кластера
    const labelText = formatFaceLabel(otherFace.person_name, otherFace.is_me, otherFace.cluster_id);
    
    const label = document.createElement('div');
    label.className = 'lb-rectangle-label';
    label.textContent = labelText;
    label.style.left = `${left}px`;
    label.style.top = `${top - 20}px`;
    label.style.position = 'absolute';
    label.style.zIndex = '10001';
    label.style.pointerEvents = 'none';
    if (otherFace.cluster_id) {
      label.style.cursor = 'pointer';
      label.onclick = function(e) {
        e.stopPropagation();
        const clusterUrl = `/face-clusters/${otherFace.cluster_id}`;
        window.open(clusterUrl, '_blank');
      };
    }
    lbBody.appendChild(label);
    drawnCount++;
  }
}

/**
 * Рисует текущее лицо (синий прямоугольник)
 * @param {Object} currentFaceBbox - Bbox текущего лица {x, y, w, h}
 * @param {Object} originalImageSize - Размеры исходного изображения {width, height, exif_orientation}
 * @param {string} personName - Имя персоны
 * @param {boolean} isMe - Флаг "это я"
 * @param {number} clusterId - ID кластера
 */
function drawFaceRectangle(currentFaceBbox, originalImageSize, personName, isMe, clusterId) {
  if (!currentFaceBbox) {
    return;
  }
  
  const lbImg = document.getElementById('lbImg');
  const lbRectangle = document.getElementById('lbRectangle');
  if (!lbImg || !lbRectangle || !lbImg.complete) {
    return;
  }
  
  // Получаем реальные размеры отображаемого изображения
  const imgRect = lbImg.getBoundingClientRect();
  const imgNaturalWidth = lbImg.naturalWidth;
  const imgNaturalHeight = lbImg.naturalHeight;
  const imgDisplayWidth = imgRect.width;
  const imgDisplayHeight = imgRect.height;
  
  if (imgNaturalWidth === 0 || imgNaturalHeight === 0) {
    return;
  }
  
  // Масштабируем bbox координаты
  let bboxX = currentFaceBbox.x;
  let bboxY = currentFaceBbox.y;
  let bboxW = currentFaceBbox.w;
  let bboxH = currentFaceBbox.h;
  
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
      const scaleX = imgNaturalWidth / originalWidth;
      const scaleY = imgNaturalHeight / originalHeight;
      bboxX = bboxX * scaleX;
      bboxY = bboxY * scaleY;
      bboxW = bboxW * scaleX;
      bboxH = bboxH * scaleY;
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
  const rectX = bboxX * scaleX;
  const rectY = bboxY * scaleY;
  const rectW = bboxW * scaleX;
  const rectH = bboxH * scaleY;
  
  // Позиционируем относительно body
  const lbBody = lbImg.parentElement;
  if (!lbBody) {
    return;
  }
  
  const bodyRect = lbBody.getBoundingClientRect();
  
  // Удаляем старую подпись для синего rectangle
  const oldLabel = lbBody.querySelector('.lb-rectangle-label.current');
  if (oldLabel) oldLabel.remove();
  
  const left = (imgRect.left - bodyRect.left) + rectX;
  const top = (imgRect.top - bodyRect.top) + rectY;
  
  // Показываем rectangle
  lbRectangle.style.left = left + 'px';
  lbRectangle.style.top = top + 'px';
  lbRectangle.style.width = rectW + 'px';
  lbRectangle.style.height = rectH + 'px';
  lbRectangle.style.display = 'block';
  
  // Добавляем подпись для синего rectangle
  const labelText = formatFaceLabel(personName, isMe, clusterId);
  
  const label = document.createElement('div');
  label.className = 'lb-rectangle-label current';
  label.textContent = labelText;
  label.style.left = `${left}px`;
  label.style.top = `${top - 20}px`;
  label.style.position = 'absolute';
  label.style.zIndex = '10002';
  label.style.pointerEvents = 'none';
  label.style.background = 'rgba(11, 87, 208, 0.95)';
  label.style.color = '#fff';
  label.style.padding = '3px 8px';
  label.style.fontSize = '12px';
  label.style.fontWeight = '700';
  label.style.borderRadius = '4px';
  label.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.3)';
  label.style.border = '1px solid rgba(11, 87, 208, 1)';
  label.style.whiteSpace = 'nowrap';
  lbBody.appendChild(label);
}
