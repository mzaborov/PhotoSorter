/**
 * Единый модуль карточки видео для просмотра и разметки кадров 1–3
 * По аналогии с photo_card.js, используется на faces.html, person_detail.html и др.
 */
(function() {
  'use strict';

  const qs = (sel, root) => (root || document).querySelector(sel);
  const qsAll = (sel, root) => Array.from((root || document).querySelectorAll(sel));

  function fetchJson(url) {
    return fetch(url).then(r => {
      if (!r.ok) throw new Error(r.statusText || String(r.status));
      return r.json();
    });
  }
  function postJson(url, data) {
    return fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    }).then(r => {
      if (!r.ok) throw new Error(r.statusText || String(r.status));
      return r.json().catch(() => ({}));
    });
  }

  let state = {
    file_id: null,
    file_path: null,
    pipeline_run_id: null,
    list_context: null,
    highlight_frame_rect: null,
    on_close: null,
    on_assign_success: null,
    frames: [], // [{frame_idx, t_sec, rects}, ...]
    activeTab: 'video', // 'video' | 1 | 2 | 3
    currentIndex: 0,
    drawEnabled: false,
    curManualRects: [],
    tempRect: null,
    dragging: false,
    dragStart: null
  };

  const modal = qs('#videoCardModal');
  const pathEl = qs('#videoCardPath');
  const copyPathBtn = qs('#videoCardCopyPath');
  const closeBtn = qs('#videoCardClose');
  const tabVideo = qs('#videoCardTabVideo');
  const tabF1 = qs('#videoCardTabF1');
  const tabF2 = qs('#videoCardTabF2');
  const tabF3 = qs('#videoCardTabF3');
  const frameActions = qs('#videoCardFrameActions');
  const imgWrap = qs('#videoCardImgWrap');
  const img = qs('#videoCardImg');
  const video = qs('#videoCardVideo');
  const canvas = qs('#videoCardCanvas');
  const keyframeRow = qs('#videoCardKeyframeRow');
  const setFrame1Btn = qs('#videoCardSetFrame1');
  const setFrame2Btn = qs('#videoCardSetFrame2');
  const setFrame3Btn = qs('#videoCardSetFrame3');
  const prevBtn = qs('#videoCardPrev');
  const nextBtn = qs('#videoCardNext');
  const positionEl = qs('#videoCardPosition');
  const rectList = qs('#videoCardRectList');

  function getPath() {
    return state.file_path || (state.list_context?.items?.[state.currentIndex]?.file_path) || '';
  }

  function getPipelineRunId() {
    return state.pipeline_run_id || state.list_context?.api_fallback?.params?.pipeline_run_id || window.pipelineRunId || null;
  }

  function frameObj(idx) {
    const arr = Array.isArray(state.frames) ? state.frames : [];
    return arr.find(x => Number(x?.frame_idx) === Number(idx)) || { frame_idx: idx, t_sec: null, rects: [] };
  }

  function buildPreviewUrl(path) {
    if (!path || !path.startsWith('local:')) return '';
    return '/api/local/preview?path=' + encodeURIComponent(path) + (getPipelineRunId() != null ? '&pipeline_run_id=' + encodeURIComponent(String(getPipelineRunId())) : '');
  }

  function buildFrameUrl(frameIdx, bustCache) {
    const path = getPath();
    if (!path) return '';
    const runId = getPipelineRunId();
    if (runId == null) return '';
    // max_dim=640 — совпадает с video_max_dim пайплайна, иначе прямоугольники смещены
    let url = '/api/faces/video-frame?pipeline_run_id=' + encodeURIComponent(String(runId)) + '&path=' + encodeURIComponent(path) + '&frame_idx=' + encodeURIComponent(String(frameIdx)) + '&max_dim=640';
    if (bustCache) url += '&_=' + Date.now();
    return url;
  }

  async function loadFrames() {
    const path = getPath();
    const runId = getPipelineRunId();
    if (!path || runId == null) return { frames: [] };
    try {
      const data = await fetchJson('/api/faces/video-manual-frames?pipeline_run_id=' + encodeURIComponent(String(runId)) + '&path=' + encodeURIComponent(path));
      state.frames = data?.frames || [];
      return data || { frames: [] };
    } catch (e) {
      state.frames = [];
      return { frames: [] };
    }
  }

  function setTabActive(tab) {
    [tabVideo, tabF1, tabF2, tabF3].forEach(b => { if (b) b.classList.toggle('active', b === tab); });
  }

  function syncFrameRectFromCurManual(rectIndex) {
    const idx = state.activeTab;
    if (idx < 1 || idx > 3) return;
    const fr = frameObj(idx);
    const rects = Array.isArray(fr.rects) ? fr.rects : [];
    const cur = state.curManualRects?.[rectIndex];
    if (!cur || rectIndex >= rects.length) return;
    rects[rectIndex].manual_person_id = cur.manual_person_id;
    rects[rectIndex].person_name = cur.person_name;
    rects[rectIndex].is_face = cur.is_face;
  }

  function renderRectList() {
    if (!rectList) return;
    rectList.innerHTML = '';
    const frames = state.frames || [];
    const allItems = [];
    for (const fr of frames) {
      const idx = Number(fr?.frame_idx) || 0;
      if (idx < 1 || idx > 3) continue;
      const rects = Array.isArray(fr.rects) ? fr.rects : [];
      rects.forEach((rect, i) => allItems.push({ frame_idx: idx, rectIndex: i, rect }));
    }
    if (allItems.length === 0) {
      rectList.innerHTML = '<span class="muted">нет</span>';
      return;
    }
    allItems.forEach(({ frame_idx, rectIndex, rect }) => {
      const pill = document.createElement('div');
      pill.className = 'rectpill';
      pill.dataset.frameIdx = String(frame_idx);
      pill.dataset.rectIndex = String(rectIndex);
      const label = rect.person_name ? escapeHtml(rect.person_name) : String(rectIndex + 1);
      pill.innerHTML = '<span class="rectpill-frame">' + frame_idx + '</span><span>' + label + '</span>';
      const actionsBtn = document.createElement('button');
      actionsBtn.className = 'rectpill-action';
      actionsBtn.textContent = '⋮';
      actionsBtn.title = 'Действия с прямоугольником';
      actionsBtn.onclick = (e) => { e.stopPropagation(); onPillActionClick(pill, actionsBtn); };
      pill.appendChild(actionsBtn);
      pill.onclick = (e) => { if (e.target.closest('.rectpill-action')) return; onPillClick(pill); };
      rectList.appendChild(pill);
    });
  }

  async function onPillClick(pill) {
    const frameIdx = parseInt(pill?.dataset?.frameIdx || '1', 10);
    const rectIndex = parseInt(pill?.dataset?.rectIndex || '0', 10);
    const needSwitch = state.activeTab === 'video' || Number(state.activeTab) !== frameIdx;
    if (needSwitch) await showFrameTab(frameIdx, true);
    showPersonSubmenuForVideo(pill, rectIndex);
  }

  async function onPillActionClick(pill, actionsBtn) {
    const frameIdx = parseInt(pill?.dataset?.frameIdx || '1', 10);
    const rectIndex = parseInt(pill?.dataset?.rectIndex || '0', 10);
    const needSwitch = state.activeTab === 'video' || Number(state.activeTab) !== frameIdx;
    if (needSwitch) await showFrameTab(frameIdx, true);
    showRectActionsMenu(rectIndex, actionsBtn);
  }

  function createRectActionsMenu(rectIndex) {
    const rect = state.curManualRects?.[rectIndex];
    if (!rect) return null;
    const menu = document.createElement('div');
    menu.className = 'rectpill-actions-menu';
    const hasPerson = !!(rect.manual_person_id || rect.person_name);
    const closeMenu = () => {
      if (menu.classList.contains('video-rect-menu')) menu.remove();
      else menu.classList.remove('open');
    };
    const outsiderBtn = document.createElement('button');
    outsiderBtn.textContent = 'Посторонний';
    outsiderBtn.onclick = async (e) => {
      e.stopPropagation();
      closeMenu();
      try {
        const data = await fetchJson('/api/persons/list');
        const outsider = (data?.persons || []).find(p => p.is_ignored === true);
        if (!outsider) { alert('Персона "Посторонний" не найдена'); return; }
        assignPersonRectIndex = rectIndex;
        applyPersonToRect(outsider.id, outsider.name || '');
        assignPersonRectIndex = null;
        await saveCurrentFrameRects();
      } catch (err) { alert(err?.message || String(err)); }
    };
    menu.appendChild(outsiderBtn);
    const personContainer = document.createElement('div');
    personContainer.style.position = 'relative';
    const personBtn = document.createElement('button');
    personBtn.textContent = hasPerson ? 'Другой' : 'Назначить персону';
    personBtn.innerHTML = '<span>' + personBtn.textContent + '</span><span style="margin-left:8px;">▶</span>';
    personBtn.style.display = 'flex';
    personBtn.style.justifyContent = 'space-between';
    personBtn.style.alignItems = 'center';
    const openSubmenu = (e) => {
      e?.stopPropagation?.();
      showPersonSubmenuForVideo(personContainer, rectIndex);
    };
    personBtn.addEventListener('mouseenter', openSubmenu);
    personBtn.addEventListener('click', openSubmenu);
    personContainer.appendChild(personBtn);
    menu.appendChild(personContainer);
    const delBtn = document.createElement('button');
    delBtn.className = 'danger';
    delBtn.textContent = 'Удалить прямоугольник';
    delBtn.onclick = async (e) => {
      e.stopPropagation();
      closeMenu();
      state.curManualRects.splice(rectIndex, 1);
      const fr = frameObj(state.activeTab);
      if (fr && Array.isArray(fr.rects) && rectIndex < fr.rects.length) fr.rects.splice(rectIndex, 1);
      renderRectList();
      drawOverlay();
      await saveCurrentFrameRects();
    };
    menu.appendChild(delBtn);
    return menu;
  }

  /**
   * Показывает подменю выбора персоны (как на карточке фото)
   */
  async function showPersonSubmenuForVideo(container, rectIndex) {
    document.querySelectorAll('.person-submenu').forEach(s => s.remove());
    if (container._submenuTimeout) {
      clearTimeout(container._submenuTimeout);
      container._submenuTimeout = null;
    }
    const runId = getPipelineRunId();
    if (runId == null) return;
    let data;
    try {
      data = await fetchJson('/api/persons/list');
    } catch (e) {
      console.error('[video_card] load persons:', e);
      return;
    }
    const persons = (data?.persons || []).filter(p => p.is_ignored !== true);
    const personsByGroup = {};
    const noGroupPersons = [];
    persons.forEach(p => {
      const group = p.group || null;
      if (group) {
        if (!personsByGroup[group]) personsByGroup[group] = [];
        personsByGroup[group].push(p);
      } else {
        noGroupPersons.push(p);
      }
    });
    const submenu = document.createElement('div');
    submenu.className = 'person-submenu';
    submenu.style.cssText = 'position:fixed;background:#fff;border:1px solid #e5e7eb;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.15);min-width:200px;max-height:400px;overflow-y:auto;z-index:10002;padding:4px 0;';
    const rect = state.curManualRects?.[rectIndex];
    const currentIsFace = rect ? (rect.is_face !== 0) : true;
    const faceToggleContainer = document.createElement('div');
    faceToggleContainer.style.cssText = 'padding:8px 14px;border-bottom:1px solid #e5e7eb;display:flex;align-items:center;gap:8px;background:#f9fafb;';
    const faceToggle = document.createElement('input');
    faceToggle.type = 'checkbox';
    faceToggle.checked = currentIsFace;
    faceToggle.style.cssText = 'width:16px;height:16px;cursor:pointer;';
    const faceToggleLabel = document.createElement('label');
    faceToggleLabel.textContent = 'Лицо';
    faceToggleLabel.style.cssText = 'font-size:12px;font-weight:600;color:#6b7280;cursor:pointer;user-select:none;flex:1;';
    faceToggleContainer.appendChild(faceToggle);
    faceToggleContainer.appendChild(faceToggleLabel);
    submenu.appendChild(faceToggleContainer);
    const groupEntries = Object.entries(personsByGroup)
      .map(([gn, ps]) => {
        const arr = ps || [];
        const order = arr.length ? Math.min(...arr.map(p => (p?.group_order != null ? Number(p.group_order) : 999))) : 999;
        return { groupName: gn, order, persons: arr };
      })
      .sort((a, b) => (a.order - b.order) || ((a.groupName || '').localeCompare(b.groupName || '', 'ru', { sensitivity: 'base' })));
    groupEntries.forEach(({ groupName, persons: ps }) => {
      const gl = document.createElement('div');
      gl.className = 'menu-group-label';
      gl.textContent = groupName;
      gl.style.cssText = 'padding:6px 14px 4px;font-size:11px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:0.5px;';
      submenu.appendChild(gl);
      [...ps].sort((a, b) => ((a?.name || '').localeCompare(b?.name || '', 'ru', { sensitivity: 'base' }))).forEach(person => {
        const btn = document.createElement('button');
        btn.textContent = (person.name || '') + (person.is_me ? ' (я)' : '');
        btn.style.cssText = 'display:block;width:100%;padding:8px 14px;text-align:left;background:none;border:none;color:#111827;cursor:pointer;font-size:13px;';
        btn.onclick = async (e) => {
          e.stopPropagation();
          submenu.remove();
          document.querySelectorAll('.rectpill-actions-menu.open, .video-rect-menu').forEach(m => { m.classList?.remove('open'); m.remove?.(); });
          const rect = state.curManualRects?.[rectIndex];
          if (!rect) return;
          rect.manual_person_id = person.id;
          rect.person_name = person.name || '';
          rect.is_face = faceToggle.checked ? 1 : 0;
          syncFrameRectFromCurManual(rectIndex);
          renderRectList();
          drawOverlay();
          await saveCurrentFrameRects();
          if (typeof state.on_assign_success === 'function') {
            try { await state.on_assign_success(getPath()); } catch (err) { console.warn('[video_card] on_assign_success:', err); }
          }
          if (typeof state.on_close === 'function') state.on_close();
        };
        submenu.appendChild(btn);
      });
    });
    noGroupPersons.sort((a, b) => ((a?.name || '').localeCompare(b?.name || '', 'ru', { sensitivity: 'base' }))).forEach(person => {
      const btn = document.createElement('button');
      btn.textContent = (person.name || '') + (person.is_me ? ' (я)' : '');
      btn.style.cssText = 'display:block;width:100%;padding:8px 14px;text-align:left;background:none;border:none;color:#111827;cursor:pointer;font-size:13px;';
      btn.onclick = async (e) => {
        e.stopPropagation();
        submenu.remove();
        document.querySelectorAll('.rectpill-actions-menu.open, .video-rect-menu').forEach(m => { m.classList?.remove('open'); m.remove?.(); });
        const rect = state.curManualRects?.[rectIndex];
        if (!rect) return;
        rect.manual_person_id = person.id;
        rect.person_name = person.name || '';
        rect.is_face = faceToggle.checked ? 1 : 0;
        syncFrameRectFromCurManual(rectIndex);
        renderRectList();
        drawOverlay();
        await saveCurrentFrameRects();
        if (typeof state.on_assign_success === 'function') {
          try { await state.on_assign_success(getPath()); } catch (err) { console.warn('[video_card] on_assign_success:', err); }
        }
        if (typeof state.on_close === 'function') state.on_close();
      };
      submenu.appendChild(btn);
    });
    document.body.appendChild(submenu);
    const cr = container.getBoundingClientRect();
    submenu.style.top = Math.max(8, Math.min(cr.top, window.innerHeight - 420)) + 'px';
    const submenuW = 220;
    if (cr.right + submenuW + 8 <= window.innerWidth) {
      submenu.style.left = (cr.right + 1) + 'px';
    } else if (cr.left - submenuW - 1 >= 0) {
      submenu.style.left = (cr.left - submenuW - 1) + 'px';
    } else {
      submenu.style.left = Math.max(8, window.innerWidth - submenuW - 8) + 'px';
    }
    const closeSubmenu = () => {
      if (submenu.parentNode) submenu.remove();
      container._submenuTimeout = null;
    };
    container.addEventListener('mouseleave', function _leave(e) {
      if (submenu.contains(e?.relatedTarget) || container.contains(e?.relatedTarget)) return;
      if (container._submenuTimeout) clearTimeout(container._submenuTimeout);
      container._submenuTimeout = setTimeout(() => {
        closeSubmenu();
        container.removeEventListener('mouseleave', _leave);
      }, 250);
    }, { once: false });
    submenu.addEventListener('mouseenter', function() {
      if (container._submenuTimeout) { clearTimeout(container._submenuTimeout); container._submenuTimeout = null; }
    });
    submenu.addEventListener('mouseleave', function _smLeave(e) {
      if (container.contains(e?.relatedTarget) || submenu.contains(e?.relatedTarget)) return;
      if (container._submenuTimeout) clearTimeout(container._submenuTimeout);
      container._submenuTimeout = setTimeout(closeSubmenu, 250);
    });
    const docClose = (e) => {
      if (!submenu.parentNode) { document.removeEventListener('click', docClose); return; }
      if (submenu.contains(e.target) || container.contains(e.target)) return;
      closeSubmenu();
      document.removeEventListener('click', docClose);
    };
    setTimeout(() => document.addEventListener('click', docClose), 50);
  }

  async function saveCurrentFrameRects() {
    const idx = state.activeTab;
    if (idx !== 1 && idx !== 2 && idx !== 3) return;
    const fr = frameObj(idx);
    try {
      await postJson('/api/faces/video-manual-frame', {
        pipeline_run_id: Number(getPipelineRunId()),
        path: getPath(),
        frame_idx: Number(idx),
        t_sec: fr?.t_sec ?? null,
        rects: state.curManualRects
      });
      showToast('Сохранено');
      loadFrames();
    } catch (e) { alert(e?.message || String(e)); }
  }

  function showRectActionsMenu(rectIndex, buttonEl) {
    document.querySelectorAll('.rectpill-actions-menu.open, .video-rect-menu').forEach(m => {
      if (m.classList) m.classList.remove('open');
      m.remove?.();
    });
    const pill = buttonEl.closest('.rectpill');
    let menu = pill?.querySelector('.rectpill-actions-menu');
    if (!menu) {
      menu = createRectActionsMenu(rectIndex);
      if (pill) {
        pill.style.position = 'relative';
        pill.appendChild(menu);
      }
    }
    menu.classList.toggle('open');
    if (pill && menu && menu.classList.contains('open')) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          const mr = menu.getBoundingClientRect();
          menu.classList.remove('rectpill-actions-menu-above-left');
          if (mr.left < 0) {
            menu.classList.add('rectpill-actions-menu-above-left');
          }
        });
      });
    }
    const close = (e) => {
      const inMenu = menu.contains(e.target);
      const inBtn = buttonEl.contains(e.target);
      if (!inMenu && !inBtn) {
        menu.classList.remove('open');
        document.removeEventListener('click', close);
      }
    };
    setTimeout(() => document.addEventListener('click', close), 10);
  }

  function getRectIndexAtPoint(px, py) {
    const rects = state.curManualRects || [];
    const scaleX = (img?.clientWidth || 1) / (img?.naturalWidth || 1);
    const scaleY = (img?.clientHeight || 1) / (img?.naturalHeight || 1);
    const nx = px / scaleX;
    const ny = py / scaleY;
    for (let i = rects.length - 1; i >= 0; i--) {
      const r = rects[i];
      if (nx >= r.x && nx <= r.x + r.w && ny >= r.y && ny <= r.y + r.h) return i;
    }
    return -1;
  }

  function showRectMenuOnCanvas(rectIndex, clientX, clientY) {
    document.querySelectorAll('.video-rect-menu').forEach(m => m.remove());
    const menu = createRectActionsMenu(rectIndex);
    if (!menu) return;
    menu.className = 'video-rect-menu';
    const wrap = imgWrap || img?.parentElement;
    if (!wrap) return;
    wrap.style.position = 'relative';
    wrap.appendChild(menu);
    menu.style.position = 'absolute';
    const wr = wrap.getBoundingClientRect();
    let leftPx = clientX - wr.left + 12;
    menu.style.left = leftPx + 'px';
    menu.style.top = (clientY - wr.top) + 'px';
    requestAnimationFrame(() => {
      const mr = menu.getBoundingClientRect();
      if (mr.right > window.innerWidth) {
        leftPx = Math.max(4, clientX - wr.left - mr.width - 4);
        menu.style.left = leftPx + 'px';
      }
    });
    const close = (e) => {
      if (!menu.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', close);
      }
    };
    setTimeout(() => document.addEventListener('click', close), 10);
  }
  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  let assignPersonRectIndex = null;

  async function assignPersonToRect(rectIndex) {
    if (rectIndex < 0 || !state.curManualRects[rectIndex]) return;
    assignPersonRectIndex = rectIndex;
    const personClear = qs('#videoCardPersonClear');
    if (personClear) personClear.style.display = '';
    const personModal = qs('#videoCardPersonModal');
    const personSelect = qs('#videoCardPersonSelect');
    if (!personModal || !personSelect) return;
    try {
      const data = await fetchJson('/api/persons/list');
      const persons = data?.persons || [];
      personSelect.innerHTML = '<option value="">— Выберите персону —</option>';
      persons.forEach(p => {
        const opt = document.createElement('option');
        opt.value = String(p.id || '');
        opt.textContent = (p.name || '') + (p.is_me ? ' (я)' : '');
        personSelect.appendChild(opt);
      });
      const rect = state.curManualRects[rectIndex];
      if (rect?.manual_person_id) {
        personSelect.value = String(rect.manual_person_id);
      } else {
        personSelect.value = '';
      }
      personModal.setAttribute('aria-hidden', 'false');
      personModal.style.display = 'flex';
    } catch (e) {
      console.error('[video_card] load persons:', e);
    }
  }

  function closePersonModal() {
    const personModal = qs('#videoCardPersonModal');
    if (personModal) {
      personModal.setAttribute('aria-hidden', 'true');
      personModal.style.display = 'none';
    }
    const personClear = qs('#videoCardPersonClear');
    const personTitle = qs('.video-card-person-title');
    if (personClear) personClear.style.display = '';
    if (personTitle) personTitle.textContent = 'Назначить персону';
    assignPersonRectIndex = null;
  }

  function applyPersonToRect(personId, personName) {
    if (assignPersonRectIndex == null) return;
    const rect = state.curManualRects[assignPersonRectIndex];
    if (!rect) return;
    if (personId) {
      rect.manual_person_id = Number(personId);
      rect.person_name = personName || '';
    } else {
      delete rect.manual_person_id;
      delete rect.person_name;
    }
    syncFrameRectFromCurManual(assignPersonRectIndex);
    renderRectList();
    drawOverlay();
  }

  function screenToNatural(pt) {
    const iw = img.naturalWidth || 1;
    const ih = img.naturalHeight || 1;
    const sx = iw / (img.clientWidth || 1);
    const sy = ih / (img.clientHeight || 1);
    return { x: pt.x * sx, y: pt.y * sy };
  }

  function getLocalPoint(e) {
    const el = canvas || img;
    if (!el) return { x: 0, y: 0 };
    const rect = el.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(rect.width, e.clientX - rect.left)),
      y: Math.max(0, Math.min(rect.height, e.clientY - rect.top))
    };
  }

  function drawOverlay() {
    if (!img || !canvas) return;
    const w = img.clientWidth || img.width || 1;
    const h = img.clientHeight || img.height || 1;
    const scaleX = w / (img.naturalWidth || 1);
    const scaleY = h / (img.naturalHeight || 1);
    canvas.width = w;
    canvas.height = h;
    canvas.style.width = w + 'px';
    canvas.style.height = h + 'px';
    canvas.style.left = '0';
    canvas.style.top = '0';
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);
    const rects = [...(state.curManualRects || []), state.tempRect].filter(Boolean);
    rects.forEach(r => {
      const sx = r.x * scaleX;
      const sy = r.y * scaleY;
      const sw = r.w * scaleX;
      const sh = r.h * scaleY;
      ctx.strokeStyle = '#3b82f6';
      ctx.lineWidth = 2;
      ctx.strokeRect(sx, sy, sw, sh);
      const label = r.person_name || '';
      if (label) {
        ctx.font = '12px system-ui, sans-serif';
        const tw = ctx.measureText(label).width + 8;
        const ly = Math.max(0, sy - 18);
        ctx.fillStyle = 'rgba(30, 64, 175, 0.9)';
        ctx.fillRect(sx, ly, tw, 18);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, sx + 4, ly + 14);
      }
    });
  }

  async function showVideoTab() {
    state.activeTab = 'video';
    setTabActive(tabVideo);
    if (keyframeRow) keyframeRow.style.display = '';
    if (img) img.style.display = 'none';
    if (video) {
      video.style.display = '';
      video.src = buildPreviewUrl(getPath());
    }
    state.curManualRects = [];
    state.drawEnabled = false;
    if (canvas) {
      canvas.classList.remove('draw-on');
      canvas.style.pointerEvents = '';
    }
    if (img) img.style.pointerEvents = '';
    renderRectList();
    drawOverlay();
  }

  async function showFrameTab(idx, bustCache) {
    state.activeTab = Number(idx);
    const tab = idx === 1 ? tabF1 : idx === 2 ? tabF2 : tabF3;
    setTabActive(tab);
    if (keyframeRow) keyframeRow.style.display = 'none';
    if (video) {
      try { video.pause(); } catch (e) {}
      video.style.display = 'none';
    }
    if (img) img.style.display = '';
    state.drawEnabled = false;
    if (canvas) {
      canvas.classList.remove('draw-on');
      canvas.style.pointerEvents = 'auto';
    }
    if (img) img.style.pointerEvents = 'none';
    if (!Array.isArray(state.frames) || state.frames.length === 0) {
      await loadFrames();
    }
    const fr = frameObj(idx);
    state.curManualRects = Array.isArray(fr.rects) ? fr.rects.map(r => {
      const base = { x: Number(r.x||0), y: Number(r.y||0), w: Number(r.w||0), h: Number(r.h||0) };
      if (r.manual_person_id != null) base.manual_person_id = Number(r.manual_person_id);
      if (r.person_name) base.person_name = String(r.person_name);
      base.is_face = (r.is_face === 0) ? 0 : 1;
      return base;
    }) : [];
    img.referrerPolicy = 'no-referrer';
    img.src = '';
    img.src = buildFrameUrl(idx, !!bustCache);
    await new Promise((res) => { if (img.complete) res(); else { img.onload = res; img.onerror = res; } });
    renderRectList();
    requestAnimationFrame(() => drawOverlay());
  }

  function closeVideoCard() {
    if (modal) {
      modal.setAttribute('aria-hidden', 'true');
      modal.style.display = 'none';
    }
    if (video) { video.pause(); video.src = ''; }
    if (img) img.src = '';
    state.frames = [];
    state.curManualRects = [];
    if (typeof state.on_close === 'function') state.on_close();
  }

  function updateNavigation() {
    const lc = state.list_context;
    if (!lc || !lc.items) {
      if (prevBtn) prevBtn.style.display = 'none';
      if (nextBtn) nextBtn.style.display = 'none';
      if (positionEl) positionEl.textContent = '—';
      return;
    }
    const total = (typeof lc.total_count === 'number' && lc.total_count > 0) ? lc.total_count : lc.items.length;
    const idx = state.currentIndex + 1;
    const atLastLoaded = state.currentIndex >= lc.items.length - 1;
    const hasMoreViaApi = lc.api_fallback && (typeof lc.total_count === 'number') && state.currentIndex < total - 1;
    const nextEnabled = !atLastLoaded || hasMoreViaApi;
    if (prevBtn) {
      prevBtn.style.display = '';
      prevBtn.disabled = state.currentIndex <= 0;
    }
    if (nextBtn) {
      nextBtn.style.display = '';
      nextBtn.disabled = !nextEnabled;
    }
    if (positionEl) positionEl.textContent = idx + ' из ' + total;
  }

  function isVideoPath(p) {
    return typeof p === 'string' && /\.(mp4|mov|avi|mkv|webm)$/i.test(p);
  }

  function goPrev() {
    const lc = state.list_context;
    if (!lc?.items || state.currentIndex <= 0) return;
    state.currentIndex--;
    const item = lc.items[state.currentIndex];
    const path = item?.file_path ?? item?.path ?? '';
    const fileId = item?.file_id ?? null;
    lc.current_index = state.currentIndex;
    if (!isVideoPath(path) && typeof window.openPhotoCard === 'function') {
      closeVideoCard();
      window.openPhotoCard({
        file_path: path || null,
        file_id: fileId,
        pipeline_run_id: getPipelineRunId(),
        list_context: lc,
        on_close: state.on_close || null
      });
    } else {
      state.file_path = path || null;
      state.file_id = fileId;
      openVideoCard({ file_path: state.file_path, file_id: state.file_id, pipeline_run_id: getPipelineRunId(), list_context: lc, current_index: state.currentIndex, on_close: state.on_close });
    }
  }

  async function goNext() {
    const lc = state.list_context;
    if (!lc?.items) return;
    const total = (typeof lc.total_count === 'number' && lc.total_count > 0) ? lc.total_count : lc.items.length;
    if (state.currentIndex >= total - 1) return;
    const atLastLoaded = state.currentIndex >= lc.items.length - 1;
    if (atLastLoaded && lc.api_fallback) {
      const pageSize = 60;
      const newIndex = state.currentIndex + 1;
      const page = Math.floor(newIndex / pageSize) + 1;
      const indexInPage = newIndex % pageSize;
      try {
        const params = new URLSearchParams(lc.api_fallback.params || {});
        params.set('page', String(page));
        params.set('page_size', String(pageSize));
        const data = await fetchJson(lc.api_fallback.endpoint + '?' + params.toString());
        if (data?.ok && data?.items && data.items[indexInPage]) {
          lc.items = data.items;
          lc.current_index = newIndex;
          state.currentIndex = newIndex;
          const item = data.items[indexInPage];
          const path = item?.file_path ?? item?.path ?? '';
          const fileId = item?.file_id ?? null;
          if (!isVideoPath(path) && typeof window.openPhotoCard === 'function') {
            closeVideoCard();
            window.openPhotoCard({
              file_path: path || null,
              file_id: fileId,
              pipeline_run_id: getPipelineRunId(),
              list_context: lc,
              on_close: state.on_close || null
            });
          } else {
            state.file_path = path || null;
            state.file_id = fileId;
            openVideoCard({ file_path: state.file_path, file_id: state.file_id, pipeline_run_id: getPipelineRunId(), list_context: lc, current_index: state.currentIndex, on_close: state.on_close });
          }
        }
      } catch (e) {
        showToast('Ошибка загрузки: ' + (e?.message || String(e)));
      }
      return;
    }
    if (state.currentIndex >= lc.items.length - 1) return;
    state.currentIndex++;
    const item = lc.items[state.currentIndex];
    const path = item?.file_path ?? item?.path ?? '';
    const fileId = item?.file_id ?? null;
    lc.current_index = state.currentIndex;
    if (!isVideoPath(path) && typeof window.openPhotoCard === 'function') {
      closeVideoCard();
      window.openPhotoCard({
        file_path: path || null,
        file_id: fileId,
        pipeline_run_id: getPipelineRunId(),
        list_context: lc,
        on_close: state.on_close || null
      });
    } else {
      state.file_path = path || null;
      state.file_id = fileId;
      openVideoCard({ file_path: state.file_path, file_id: state.file_id, pipeline_run_id: getPipelineRunId(), list_context: lc, current_index: state.currentIndex, on_close: state.on_close });
    }
  }

  /**
   * Открывает карточку видео
   * @param {Object} options
   * @param {number|null} options.file_id
   * @param {string|null} options.file_path - обязателен для видео
   * @param {number|null} options.pipeline_run_id
   * @param {Object|null} options.list_context - { items, current_index, total_count, api_fallback }
   * @param {Object|null} options.highlight_frame_rect - { frame_idx, rect_index }
   * @param {Function|null} options.on_close
   */
  function openVideoCard(options) {
    if (!options) return;
    state = {
      file_id: options.file_id || null,
      file_path: options.file_path || null,
      pipeline_run_id: options.pipeline_run_id ?? getPipelineRunId(),
      list_context: options.list_context || null,
      highlight_frame_rect: options.highlight_frame_rect || null,
      on_close: options.on_close || null,
      frames: [],
      activeTab: 'video',
      currentIndex: (options.list_context?.current_index ?? options.current_index ?? 0),
      drawEnabled: false,
      curManualRects: [],
      tempRect: null,
      dragging: false,
      dragStart: null
    };
    if (state.list_context?.items) {
      const item = state.list_context.items[state.currentIndex];
      if (item) {
        state.file_id = item.file_id ?? state.file_id;
        state.file_path = item.file_path ?? item.path ?? state.file_path;
      }
    }
    if (!state.file_path) {
      console.error('[video_card] file_path is required');
      return;
    }
    const path = state.file_path;
    if (/^disk/i.test(path)) {
      const diskPath = path.replace(/^disk:?[/\\]?/i, 'disk:/').replace(/\\/g, '/').replace(/^disk:\/+/, 'disk:/');
      const yadiskUrl = 'https://disk.yandex.ru/client/disk?path=' + encodeURIComponent(diskPath);
      pathEl.innerHTML = '<a href="' + yadiskUrl.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/"/g, '&quot;') + '" target="_blank" rel="noopener noreferrer" style="color:inherit;text-decoration:underline;">' + path.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;') + '</a>';
    } else {
      pathEl.textContent = path;
    }
    modal.setAttribute('aria-hidden', 'false');
    modal.style.display = 'flex';
    const specialActions = qs('#videoCardSpecialActions');
    if (specialActions) {
      specialActions.style.display = getPipelineRunId() != null ? 'flex' : 'none';
    }
    updateNavigation();
    updateUndoButton();
    loadGroupsAndPersons();
    const tb = qs('#videoCardToggleRectangles');
    if (tb) tb.textContent = 'Скрыть прямоугольники';
    rectsVisible = true;
    if (canvas) canvas.style.visibility = '';
    const hl = state.highlight_frame_rect;
    // Сначала загружаем кадры, затем показываем вкладку — иначе rects не подставляются
    loadFrames().then((data) => {
      const frames = data?.frames || state.frames || [];
      if (hl && hl.frame_idx >= 1 && hl.frame_idx <= 3) {
        showFrameTab(hl.frame_idx);
      } else {
        const firstWithRects = frames.find(f => Array.isArray(f?.rects) && f.rects.length > 0);
        if (firstWithRects && firstWithRects.frame_idx >= 1 && firstWithRects.frame_idx <= 3) {
          showFrameTab(firstWithRects.frame_idx);
        } else {
          showFrameTab(1);
        }
      }
    }).catch(() => {
      showFrameTab(1);
    });
  }

  function showToast(msg) {
    if (typeof window.showToast === 'function') {
      window.showToast(msg);
      return;
    }
    const t = document.createElement('div');
    t.textContent = msg;
    t.style.cssText = 'position:fixed;bottom:20px;left:50%;transform:translateX(-50%);background:#111;color:#fff;padding:10px 20px;border-radius:8px;z-index:10002;font-size:14px;';
    document.body.appendChild(t);
    setTimeout(() => t.remove(), 2000);
  }

  /** Заполняет select групп как на faces (Поездки, остальные, + Создать новую группу). */
  function fillGroupSelect(groupSelect, groups) {
    const predefined = ['Здоровье', 'Чеки', 'Дом и ремонт', 'Артефакты людей'];
    const tripsKeywords = ['Турция', 'Минск', 'Италия', 'Испания', 'Греция', 'Франция', 'Польша', 'Чехия', 'Германия', 'Тургояк'];
    const allGroups = new Set();
    const groupsWithData = [];
    (groups || []).forEach(g => {
      const name = g.group_path || '';
      if (!name) return;
      allGroups.add(name);
      const hasYear = /\d{4}/.test(name);
      const hasPlace = tripsKeywords.some(kw => name.includes(kw));
      const startsWithYear = /^\d{4}\s/.test(name);
      const isTrip = (hasYear && hasPlace) || (startsWithYear && name.length > 5);
      groupsWithData.push({ name, isTrip, last_created_at: g.last_created_at || '' });
    });
    predefined.forEach(p => {
      if (!allGroups.has(p)) groupsWithData.push({ name: p, isTrip: false, last_created_at: '' });
    });
    const categoryMap = {};
    const otherGroups = [];
    groupsWithData.forEach(gd => {
      if (gd.isTrip) {
        if (!categoryMap['Поездки']) categoryMap['Поездки'] = [];
        categoryMap['Поездки'].push({ name: gd.name, last_created_at: gd.last_created_at });
      } else {
        otherGroups.push(gd.name);
      }
    });
    groupSelect.innerHTML = '<option value="">Назначить группу...</option>';
    const tripsOptgroup = document.createElement('optgroup');
    tripsOptgroup.label = 'Поездки';
    if (categoryMap['Поездки'] && categoryMap['Поездки'].length > 0) {
      categoryMap['Поездки'].sort((a, b) => (b.last_created_at || '').localeCompare(a.last_created_at || ''));
      categoryMap['Поездки'].forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.name;
        opt.textContent = t.name;
        tripsOptgroup.appendChild(opt);
      });
    }
    const createTrip = document.createElement('option');
    createTrip.value = '__create_new_Поездки__';
    createTrip.textContent = '+ Создать новую поездку...';
    createTrip.style.fontStyle = 'italic';
    tripsOptgroup.appendChild(createTrip);
    groupSelect.appendChild(tripsOptgroup);
    otherGroups.sort();
    otherGroups.forEach(path => {
      const opt = document.createElement('option');
      opt.value = path;
      opt.textContent = path;
      groupSelect.appendChild(opt);
    });
    const createOpt = document.createElement('option');
    createOpt.value = '__create_new__';
    createOpt.textContent = '+ Создать новую группу...';
    createOpt.style.fontStyle = 'italic';
    groupSelect.appendChild(createOpt);
  }

  async function loadGroupsAndPersons() {
    const runId = getPipelineRunId();
    if (runId == null) return;
    const groupSelect = qs('#videoCardGroupSelect');
    const personDropdown = qs('#videoCardPersonDropdown');
    if (groupSelect) {
      try {
        const data = await fetchJson('/api/faces/groups-with-files?pipeline_run_id=' + encodeURIComponent(String(runId)));
        const groups = data?.groups || [];
        fillGroupSelect(groupSelect, groups);
      } catch (e) { console.error('[video_card] load groups:', e); }
    }
    if (personDropdown) {
      try {
        const data = await fetchJson('/api/persons/list');
        const persons = data?.persons || [];
        personDropdown.innerHTML = '<option value="">— Выберите персону —</option>';
        persons.forEach(p => {
          const opt = document.createElement('option');
          opt.value = String(p.id || '');
          opt.textContent = (p.name || '') + (p.is_me ? ' (я)' : '');
          personDropdown.appendChild(opt);
        });
      } catch (e) { console.error('[video_card] load persons:', e); }
    }
  }

  if (copyPathBtn) {
    copyPathBtn.onclick = async () => {
      const path = getPath();
      if (!path) return;
      const orig = copyPathBtn.innerHTML;
      try {
        await navigator.clipboard.writeText(path);
        copyPathBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block!important;width:14px!important;height:14px!important;"><path d="M13 4L6 11L3 8" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
        setTimeout(() => { copyPathBtn.innerHTML = orig; }, 1000);
      } catch (e) {
        try {
          const ta = document.createElement('textarea');
          ta.value = path;
          ta.style.cssText = 'position:fixed;opacity:0;';
          document.body.appendChild(ta);
          ta.select();
          document.execCommand('copy');
          document.body.removeChild(ta);
          copyPathBtn.innerHTML = '<svg width="14" height="14" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg" style="display:block!important;width:14px!important;height:14px!important;"><path d="M13 4L6 11L3 8" stroke="#22c55e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none"/></svg>';
          setTimeout(() => { copyPathBtn.innerHTML = orig; }, 1000);
        } catch (e2) {
          alert('Не удалось скопировать путь: ' + (e?.message || e));
        }
      }
    };
  }
  if (closeBtn) closeBtn.onclick = closeVideoCard;

  let undoStack = [];
  const MAX_UNDO_STACK_SIZE = 20;

  function pushUndoAction(action) {
    undoStack.push(action);
    if (undoStack.length > MAX_UNDO_STACK_SIZE) undoStack.shift();
  }

  function updateUndoButton() {
    const undoBtn = qs('#videoCardUndo');
    if (undoBtn) {
      undoBtn.disabled = undoStack.length === 0;
      undoBtn.title = undoStack.length > 0 ? 'Отменить последнее действие' : 'Нет действий для отмены';
    }
  }

  async function undoLastAction() {
    if (undoStack.length === 0) {
      showToast('Нет действий для отмены');
      return;
    }
    const action = undoStack.pop();
    updateUndoButton();
    if (action.type !== 'delete_file') return;
    if (!action.undo_data?.delete_path || !action.undo_data?.original_path || !action.pipeline_run_id) {
      showToast('Ошибка: нет данных для отмены');
      return;
    }
    try {
      const res = await fetch('/api/faces/restore-from-delete', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          pipeline_run_id: action.pipeline_run_id,
          delete_path: action.undo_data.delete_path,
          original_path: action.undo_data.original_path,
          original_name: action.undo_data.original_name ?? '',
          original_parent_path: action.undo_data.original_parent_path ?? ''
        })
      });
      if (res.ok) {
        showToast('Файл восстановлен');
        if (typeof state.on_close === 'function') state.on_close();
      } else {
        const err = await res.json().catch(() => ({}));
        alert('Ошибка восстановления: ' + (err.detail || res.statusText));
      }
    } catch (e) {
      alert('Ошибка: ' + (e?.message || String(e)));
      undoStack.push(action);
      updateUndoButton();
    }
  }

  const deleteBtn = qs('#videoCardDelete');
  if (deleteBtn) {
    deleteBtn.onclick = async () => {
      const path = getPath();
      const runId = getPipelineRunId();
      if (!path || runId == null) return;
      try {
        const data = await postJson('/api/faces/delete', { pipeline_run_id: Number(runId), path });
        if (data?.undo_data) {
          pushUndoAction({ type: 'delete_file', undo_data: data.undo_data, pipeline_run_id: Number(runId) });
          updateUndoButton();
        }
        showToast('Файл удалён');
        const lc = state.list_context;
        if (lc?.items && state.currentIndex < lc.items.length - 1) {
          goNext();
        } else {
          closeVideoCard();
        }
      } catch (e) {
        alert(e?.message || String(e));
      }
    };
  }

  const undoBtn = qs('#videoCardUndo');
  if (undoBtn) undoBtn.onclick = undoLastAction;

  const assignOutsiderBtn = qs('#videoCardAssignOutsider');
  if (assignOutsiderBtn) {
    assignOutsiderBtn.onclick = async () => {
      const path = getPath();
      const runId = getPipelineRunId();
      if (!path) { alert('Нет пути к файлу'); return; }
      try {
        const payload = { path };
        if (state.file_id) payload.file_id = state.file_id;
        if (runId != null) payload.pipeline_run_id = Number(runId);
        const res = await postJson('/api/faces/rectangles/assign-outsider', payload);
        showToast(res.assigned_count != null ? `Назначено посторонними: ${res.assigned_count}` : 'Готово');
        await loadFrames();
        if (typeof state.on_close === 'function') state.on_close();
      } catch (e) {
        alert(e?.message || String(e));
      }
    };
  }

  const groupSelectEl = qs('#videoCardGroupSelect');
  if (groupSelectEl) {
    groupSelectEl.onchange = async () => {
      let groupPath = groupSelectEl.value;
      if (!groupPath) return;
      const path = getPath();
      const runId = getPipelineRunId();
      if (!path || runId == null) return;
      const origVal = groupSelectEl.value;
      if (groupPath === '__create_new__' || groupPath === '__create_new_Поездки__') {
        const isTrips = groupPath === '__create_new_Поездки__';
        const promptText = isTrips ? 'Введите название поездки (например: 2025 Италия):' : 'Введите название новой группы:';
        const newName = prompt(promptText);
        if (!newName || !newName.trim()) {
          groupSelectEl.value = '';
          return;
        }
        groupPath = newName.trim();
      }
      try {
        await postJson('/api/faces/assign-group', { pipeline_run_id: Number(runId), path, group_path: groupPath });
        showToast('Группа назначена');
        groupSelectEl.value = '';
        if (typeof state.on_close === 'function') state.on_close();
      } catch (e) {
        alert(e?.message || String(e));
        groupSelectEl.value = origVal;
      }
    };
  }

  function showAssignPersonMenu(container) {
    document.querySelectorAll('.assign-person-menu').forEach(menu => menu.remove());
    const menu = document.createElement('div');
    menu.className = 'assign-person-menu';
    menu.style.position = 'absolute';
    menu.style.left = '0';
    menu.style.bottom = '100%';
    menu.style.marginBottom = '4px';
    menu.style.background = '#fff';
    menu.style.border = '1px solid #e5e7eb';
    menu.style.borderRadius = '8px';
    menu.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
    menu.style.minWidth = '200px';
    menu.style.zIndex = '10001';
    menu.style.padding = '4px 0';

    const hasRects = (state.curManualRects && state.curManualRects.length > 0) || (state.frames && state.frames.some(f => (f.rects || []).length > 0));

    const areaContainer = document.createElement('div');
    areaContainer.style.position = 'relative';
    const areaBtn = document.createElement('button');
    areaBtn.innerHTML = '<span>указать область на кадре</span><span style="margin-left: 8px;">▶</span>';
    areaBtn.style.cssText = 'width:100%;padding:10px 12px;text-align:left;border:none;background:#fff;cursor:pointer;font-size:14px;display:flex;justify-content:space-between;align-items:center;';
    areaBtn.addEventListener('mouseenter', () => showAreaSubmenu(areaContainer));
    areaContainer.appendChild(areaBtn);
    menu.appendChild(areaContainer);

    const wholeBtn = document.createElement('button');
    wholeBtn.textContent = 'к видео целиком';
    wholeBtn.style.cssText = 'width:100%;padding:10px 12px;text-align:left;border:none;background:#fff;cursor:pointer;font-size:14px;';
    wholeBtn.style.cursor = hasRects ? 'not-allowed' : 'pointer';
    wholeBtn.style.opacity = hasRects ? '0.5' : '1';
    wholeBtn.disabled = hasRects;
    if (!hasRects) {
      wholeBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        menu.remove();
        showPersonDialogForWholeVideo();
      });
    }
    menu.appendChild(wholeBtn);

    const closeMenu = (e) => {
      if (!menu.contains(e.target) && !container.contains(e.target)) {
        menu.remove();
        document.removeEventListener('click', closeMenu);
      }
    };
    setTimeout(() => document.addEventListener('click', closeMenu), 100);
    container.appendChild(menu);
  }

  function showAreaSubmenu(container) {
    document.querySelectorAll('.area-submenu').forEach(s => s.remove());
    const submenu = document.createElement('div');
    submenu.className = 'area-submenu';
    submenu.style.cssText = 'position:absolute;left:100%;top:0;margin-left:4px;background:#fff;border:1px solid #e5e7eb;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.15);min-width:150px;z-index:10002;padding:4px 0;';
    ['лицо', 'без лица'].forEach(label => {
      const btn = document.createElement('button');
      btn.textContent = label;
      btn.style.cssText = 'width:100%;padding:10px 12px;text-align:left;border:none;background:#fff;cursor:pointer;font-size:14px;';
      btn.addEventListener('click', async (e) => {
        e.stopPropagation();
        submenu.remove();
        document.querySelectorAll('.assign-person-menu').forEach(m => m.remove());
        await enableAnnotateMode();
      });
      submenu.appendChild(btn);
    });
    container.appendChild(submenu);
  }

  async function enableAnnotateMode() {
    if (state.activeTab === 'video' || (state.activeTab !== 1 && state.activeTab !== 2 && state.activeTab !== 3)) {
      await showFrameTab(1);
    }
    state.drawEnabled = true;
    if (canvas) { canvas.classList.add('draw-on'); canvas.style.pointerEvents = 'auto'; }
    if (img) img.style.pointerEvents = 'none';
    drawOverlay();
  }

  function exitDrawMode() {
    state.drawEnabled = false;
    if (canvas) { canvas.classList.remove('draw-on'); canvas.style.pointerEvents = 'auto'; }
    if (img) img.style.pointerEvents = 'none';
    drawOverlay();
  }

  async function showPersonDialogForWholeVideo() {
    assignPersonRectIndex = -1;
    const personModal = qs('#videoCardPersonModal');
    const personSelect = qs('#videoCardPersonSelect');
    const personClear = qs('#videoCardPersonClear');
    const personTitle = qs('.video-card-person-title');
    if (!personModal || !personSelect) return;
    try {
      const data = await fetchJson('/api/persons/list');
      const persons = (data?.persons || []).filter(p => p.is_ignored !== true);
      personSelect.innerHTML = '<option value="">— Выберите персону —</option>';
      persons.forEach(p => {
        const opt = document.createElement('option');
        opt.value = String(p.id || '');
        opt.textContent = (p.name || '') + (p.is_me ? ' (я)' : '');
        personSelect.appendChild(opt);
      });
      if (personTitle) personTitle.textContent = 'Выберите персону для видео';
      if (personClear) personClear.style.display = 'none';
      personModal.setAttribute('aria-hidden', 'false');
      personModal.style.display = 'flex';
      personSelect.value = '';
    } catch (e) { console.error('[video_card] load persons:', e); }
  }

  const assignPersonBtn = qs('#videoCardAssignPerson');
  const assignPersonContainer = qs('#videoCardAssignPersonContainer');
  if (assignPersonBtn && assignPersonContainer) {
    assignPersonBtn.onclick = (e) => {
      const existing = assignPersonContainer.querySelector('.assign-person-menu');
      if (existing) {
        existing.remove();
      } else {
        showAssignPersonMenu(assignPersonContainer);
      }
      e.stopPropagation();
    };
  }

  const toggleRectBtn = qs('#videoCardToggleRectangles');
  let rectsVisible = true;
  if (toggleRectBtn && canvas) {
    toggleRectBtn.onclick = () => {
      rectsVisible = !rectsVisible;
      toggleRectBtn.textContent = rectsVisible ? 'Скрыть прямоугольники' : 'Показать прямоугольники';
      if (canvas) canvas.style.visibility = rectsVisible ? '' : 'hidden';
    };
  }

  const personOkBtn = qs('#videoCardPersonOk');
  const personClearBtn = qs('#videoCardPersonClear');
  const personCancelBtn = qs('#videoCardPersonCancel');
  const personSelectEl = qs('#videoCardPersonSelect');
  if (personOkBtn && personSelectEl) {
    personOkBtn.onclick = async () => {
      const val = personSelectEl.value;
      if (assignPersonRectIndex === -1) {
        if (!val) return;
        const path = getPath();
        const runId = getPipelineRunId();
        if (!path || runId == null) return;
        try {
          await postJson('/api/persons/assign-file', { pipeline_run_id: Number(runId), file_path: path, person_id: parseInt(val) });
          showToast('Персона назначена');
          closePersonModal();
          if (typeof state.on_close === 'function') state.on_close();
        } catch (e) { alert(e?.message || String(e)); }
      } else {
        const opt = personSelectEl.options[personSelectEl.selectedIndex];
        const name = opt ? opt.textContent : '';
        applyPersonToRect(val ? Number(val) : null, val ? (name || '') : null);
        closePersonModal();
      }
    };
  }
  if (personClearBtn) {
    personClearBtn.onclick = () => {
      applyPersonToRect(null, null);
      closePersonModal();
    };
  }
  if (personCancelBtn) personCancelBtn.onclick = closePersonModal;
  if (tabVideo) tabVideo.onclick = () => showVideoTab();
  if (tabF1) tabF1.onclick = () => showFrameTab(1, true);
  if (tabF2) tabF2.onclick = () => showFrameTab(2, true);
  if (tabF3) tabF3.onclick = () => showFrameTab(3, true);


  // Рисование и клик по прямоугольнику на canvas
  if (canvas) {
    canvas.onclick = (e) => {
      if (state.drawEnabled) return;
      const pt = getLocalPoint(e);
      const idx = getRectIndexAtPoint(pt.x, pt.y);
      if (idx >= 0) showRectMenuOnCanvas(idx, e.clientX, e.clientY);
    };
    canvas.onmousedown = (e) => {
      if (!state.drawEnabled) return;
      e.preventDefault();
      state.dragging = true;
      state.dragStart = getLocalPoint(e);
      state.tempRect = null;
    };
    canvas.onmousemove = (e) => {
      if (!state.drawEnabled || !state.dragging || !state.dragStart) return;
      e.preventDefault();
      const p = getLocalPoint(e);
      const x0 = Math.min(state.dragStart.x, p.x);
      const y0 = Math.min(state.dragStart.y, p.y);
      const w0 = Math.abs(state.dragStart.x - p.x);
      const h0 = Math.abs(state.dragStart.y - p.y);
      const n0 = screenToNatural({ x: x0, y: y0 });
      const n1 = screenToNatural({ x: x0 + w0, y: y0 + h0 });
      state.tempRect = { x: Math.round(n0.x), y: Math.round(n0.y), w: Math.round(n1.x - n0.x), h: Math.round(n1.y - n0.y) };
      drawOverlay();
    };
    canvas.onmouseup = async (e) => {
      if (!state.drawEnabled || !state.dragging) return;
      e.preventDefault();
      state.dragging = false;
      if (state.tempRect && state.tempRect.w >= 8 && state.tempRect.h >= 8) {
        const r = { ...state.tempRect, is_face: 1 };
        state.curManualRects.push(r);
        state.tempRect = null;
        renderRectList();
        exitDrawMode();
        try {
          await saveCurrentFrameRects();
          if (typeof state.on_close === 'function') state.on_close();
        } catch (err) {
          alert(err?.message || String(err));
        }
      } else {
        state.tempRect = null;
      }
      drawOverlay();
    };
  }

  // Кнопки «На Кадр 1/2/3» при паузе: устанавливают ключевой кадр и сбрасывают привязки для него в БД
  const setKeyframe = async (idx) => {
    if (!video) return;
    const t = video.currentTime;
    const path = getPath();
    const runId = getPipelineRunId();
    if (!path || runId == null) {
      showToast('Нет пути или прогона');
      return;
    }
    try {
      const data = await postJson('/api/faces/video-keyframe-position', {
        pipeline_run_id: Number(runId),
        path,
        frame_idx: idx,
        t_sec: t,
      });
      showToast(`Кадр ${idx} установлен`);
      await loadFrames();
      showFrameTab(idx, true);
      if (typeof state.on_close === 'function') state.on_close();
    } catch (e) {
      showToast(e?.message || 'Ошибка');
    }
  };
  if (setFrame1Btn) setFrame1Btn.onclick = () => setKeyframe(1);
  if (setFrame2Btn) setFrame2Btn.onclick = () => setKeyframe(2);
  if (setFrame3Btn) setFrame3Btn.onclick = () => setKeyframe(3);

  if (prevBtn) prevBtn.onclick = goPrev;
  if (nextBtn) nextBtn.onclick = goNext;

  // Экспорт для глобального доступа
  window.openVideoCard = openVideoCard;
  window.closeVideoCard = closeVideoCard;
})();
