/* Review actual legend crops before rerunning detection. Coordinates are inclusive
   pixels of the saved job image, independent of the browser's display scale. */
(function (root) {
  'use strict';

  const clone = value => JSON.parse(JSON.stringify(value));
  const boxCenter = box => [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2];
  const contains = (box, x, y) => x >= box[0] && x <= box[2] && y >= box[1] && y <= box[3];
  const entryCenter = entry => Array.isArray(entry.center) && entry.center.length === 2 &&
    entry.center.every(Number.isFinite) && contains(entry.box, ...entry.center) ? entry.center.slice() : boxCenter(entry.box);
  const unavailable = entry => /fail|unavailable/i.test(entry.status || '');
  function resizedBox(box, width, height, bounds) {
    const center = boxCenter(box);
    width = Math.min(Math.max(3, Math.round(width)), bounds[2] - bounds[0] + 1);
    height = Math.min(Math.max(3, Math.round(height)), bounds[3] - bounds[1] + 1);
    const x = Math.max(bounds[0], Math.min(Math.round(center[0] - (width - 1) / 2), bounds[2] - width + 1));
    const y = Math.max(bounds[1], Math.min(Math.round(center[1] - (height - 1) / 2), bounds[3] - height + 1));
    return [x, y, x + width - 1, y + height - 1];
  }
  function canvasPoint(clientX, clientY, rect, view) {
    return [Math.round(view[0] + (clientX - rect.left) / rect.width * (view[2] - view[0] + 1)),
      Math.round(view[1] + (clientY - rect.top) / rect.height * (view[3] - view[1] + 1))];
  }
  function installStyle() {
    if (document.getElementById('legendReviewV46Style')) return;
    const style = document.createElement('style');
    style.id = 'legendReviewV46Style';
    style.textContent = `
      .lr-review{border:1px solid #bdd3df;border-radius:10px;padding:14px;margin:0 0 16px;background:#f8fbfd;color:#183343;font-size:14px}
      .lr-review h3{margin:0 0 7px;font-size:17px}.lr-review p{margin:6px 0;line-height:1.5}
      .lr-review button{border:1px solid #a5b9c5;border-radius:6px;background:white;color:#1c3d50;padding:7px 10px;cursor:pointer;font:inherit}
      .lr-review button:disabled{opacity:.5;cursor:default}.lr-review button[aria-pressed=true]{background:#12647a;color:white;border-color:#12647a}
      .lr-review .lr-primary{background:#12647a;color:white;border-color:#12647a;font-weight:600}
      .lr-tools,.lr-footer{display:flex;flex-wrap:wrap;gap:7px;align-items:center;margin:9px 0}.lr-footer{border-top:1px solid #dce7ed;padding-top:10px;margin-bottom:0}
      .lr-layout{display:grid;grid-template-columns:minmax(0,1fr) 210px;gap:12px;align-items:start}
      .lr-canvas-wrap{overflow:auto;border:1px solid #ccd9df;border-radius:5px;background:white}
      .lr-canvas{display:block;max-width:100%;height:auto;cursor:crosshair;touch-action:manipulation}
      .lr-list{max-height:310px;overflow:auto;display:flex;flex-direction:column;gap:5px;padding:2px}
      .lr-list button{display:flex;align-items:center;gap:7px;text-align:left;min-width:0;width:100%;padding:5px}
      .lr-list button[aria-pressed=true]{background:#e3f3f7;color:#173743;border:2px solid #12728b;padding:4px}
      .lr-list canvas{width:56px;height:38px;object-fit:contain;background:white;border:1px solid #d9e0e3;flex-shrink:0}
      .lr-list span{overflow-wrap:anywhere;white-space:pre-line;font-size:13px}.lr-list .lr-entry-failed{color:#a32724}.lr-crop{padding-top:8px;display:flex;flex-wrap:wrap;gap:7px;align-items:center}
      .lr-crop input{width:65px;padding:4px;font:inherit}.lr-crop label{font-size:12px}.lr-count{font-weight:600}
      .lr-muted{font-size:12px;color:#516676}.lr-status{white-space:pre-line;font-size:13px;min-height:1.5em}.lr-error{color:#a32724}
      .lr-review :focus-visible{outline:3px solid #e5a52b;outline-offset:2px}
      @media(max-width:720px){.lr-layout{grid-template-columns:1fr}.lr-list{max-height:170px;display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr))}}
    `;
    document.head.appendChild(style);
  }
  function element(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  async function readJSON(url, options) {
    const response = await fetch(url, {cache: 'no-store', ...options});
    const value = await response.json().catch(() => null);
    if (!response.ok) {
      const detail = value?.detail || value?.message;
      throw new Error(typeof detail === 'string' ? detail : 'Legend review request failed (' + response.status + ').');
    }
    return value;
  }

  function mount(host, options) {
    installStyle();
    const data = options.data || {};
    const state = {alive: true, loaded: false, loading: null, pending: false, review: null,
      entries: [], initial: [], undo: [], selected: null, mode: 'select', picture: null, view: null};
    const current = () => state.alive && (!options.isCurrent || options.isCurrent());
    const busy = () => !current() || state.pending || !!options.isBusy?.();
    const panel = element('section', 'lr-review');
    panel.setAttribute('aria-label', 'Review legend swatches');
    panel.append(element('h3', '', 'Review legend swatches'));
    panel.append(element('p', '', 'Check each outlined swatch against the original legend. Add a missing swatch by clicking its center, or select an incorrect swatch and delete it.'));
    const status = element('p', 'lr-status', 'Open this panel to load the original legend.');
    status.setAttribute('role', 'status'); status.setAttribute('aria-live', 'polite');
    const contents = element('div'); contents.hidden = true;
    const tools = element('div', 'lr-tools'); tools.setAttribute('role', 'group'); tools.setAttribute('aria-label', 'Legend editing tools');
    const button = (label, action, className = '') => {
      const node = element('button', className, label); node.type = 'button'; node.addEventListener('click', action); return node;
    };
    const select = button('Select swatch', () => setMode('select'));
    const add = button('Add by center', () => setMode('add'));
    const move = button('Move selected center', () => setMode('move'));
    const remove = button('Delete selected', deleteSelected);
    const undo = button('Undo', undoEdit);
    tools.append(select, add, move, remove, undo);
    const count = element('p', 'lr-count');
    const layout = element('div', 'lr-layout');
    const imageColumn = element('div');
    const canvasWrap = element('div', 'lr-canvas-wrap');
    const canvas = element('canvas', 'lr-canvas');
    canvas.setAttribute('aria-label', 'Original legend with extracted swatch boxes; choose Add by center then click a missing symbol');
    canvas.tabIndex = 0; canvasWrap.append(canvas); imageColumn.append(canvasWrap);
    const legendKey = element('p', 'lr-muted', 'Green: extracted swatch · Red: extraction needs review · Amber: added or adjusted · Blue: selected');
    imageColumn.append(legendKey);
    const crop = element('div', 'lr-crop');
    const selectedLabel = element('span', '', 'Select a swatch to adjust its crop.');
    const width = element('input'), height = element('input');
    for (const input of [width, height]) { input.type = 'number'; input.min = '3'; input.step = '1'; }
    width.setAttribute('aria-label', 'Selected swatch crop width in pixels'); height.setAttribute('aria-label', 'Selected swatch crop height in pixels');
    const widthLabel = element('label', '', 'Width (px) '); widthLabel.append(width);
    const heightLabel = element('label', '', 'Height (px) '); heightLabel.append(height);
    const resize = button('Resize crop', resizeSelected);
    crop.append(selectedLabel, widthLabel, heightLabel, resize);
    const failureDetail = element('p', 'lr-status lr-error'); failureDetail.hidden = true;
    imageColumn.append(crop, failureDetail, element('p', 'lr-muted', 'Include the whole symbol and its short line segment; avoid neighboring text. Resizing changes the crop, not the source image.'));
    const list = element('div', 'lr-list'); list.setAttribute('aria-label', 'Extracted swatches');
    layout.append(imageColumn, list);
    const footer = element('div', 'lr-footer');
    const apply = button('Apply swatches & rerun detection', rerun, 'lr-primary');
    const reset = button('Reset changes', resetEdits);
    footer.append(apply, reset);
    contents.append(count, tools, layout, footer, element('p', 'lr-muted', 'These changes take effect after rerunning detection. Existing point edits are not transferred to the new detection result.'));
    panel.append(contents, status); host.replaceChildren(panel);

    function message(text, error = false) { status.textContent = text; status.classList.toggle('lr-error', error); }
    function selectedEntry() { return state.entries.find(entry => entry.id === state.selected); }
    function changed() { return JSON.stringify(state.entries) !== JSON.stringify(state.initial); }
    function snapshot() { state.undo.push({entries: clone(state.entries), selected: state.selected}); if (state.undo.length > 40) state.undo.shift(); }
    function setMode(mode) {
      if (busy()) return;
      state.mode = mode;
      message(mode === 'add' ? 'Click the center of a missing swatch in the original legend.'
        : mode === 'move' ? 'Click the correct center for the selected swatch.'
          : 'Select an outlined swatch or its crop in the list.');
      syncBusy();
    }
    function syncBusy() {
      const locked = busy() || !state.loaded;
      for (const control of panel.querySelectorAll('button,input')) control.disabled = locked;
      const selected = !!selectedEntry();
      move.disabled = locked || !selected; remove.disabled = locked || !selected;
      resize.disabled = locked || !selected; width.disabled = locked || !selected; height.disabled = locked || !selected;
      undo.disabled = locked || !state.undo.length; reset.disabled = locked || !changed();
      apply.disabled = locked || !changed() || !state.entries.length;
      select.setAttribute('aria-pressed', String(state.mode === 'select'));
      add.setAttribute('aria-pressed', String(state.mode === 'add'));
      move.setAttribute('aria-pressed', String(state.mode === 'move'));
      canvas.style.cursor = locked ? 'wait' : state.mode === 'select' ? 'pointer' : 'crosshair';
    }
    function render() {
      if (!state.loaded || !current()) return;
      count.textContent = state.entries.length + (state.entries.length === 1 ? ' swatch' : ' swatches') + (changed() ? ' · Changes not yet applied' : '');
      const selected = selectedEntry();
      selectedLabel.textContent = selected ? 'Swatch ' + (state.entries.indexOf(selected) + 1) + ' crop:' : 'Select a swatch to adjust its crop.';
      width.value = selected ? selected.box[2] - selected.box[0] + 1 : '';
      height.value = selected ? selected.box[3] - selected.box[1] + 1 : '';
      failureDetail.hidden = !selected || !unavailable(selected);
      failureDetail.textContent = selected && unavailable(selected)
        ? 'Extraction failed: ' + (selected.reason || 'A usable marker template could not be extracted from this crop.') + ' Adjust the crop or delete this entry, then rerun detection.' : '';
      width.max = state.review.legend_box[2] - state.review.legend_box[0] + 1;
      height.max = state.review.legend_box[3] - state.review.legend_box[1] + 1;
      draw(); renderList(); syncBusy();
    }
    function draw() {
      const ctx = canvas.getContext('2d'), view = state.view;
      const vw = view[2] - view[0] + 1, vh = view[3] - view[1] + 1;
      const scale = canvas.width / vw;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(state.picture, view[0], view[1], vw, vh, 0, 0, canvas.width, canvas.height);
      state.entries.forEach((entry, index) => {
        const b = entry.box, chosen = entry.id === state.selected;
        const color = unavailable(entry) ? '#b33535' : chosen ? '#087bba' : entry.source === 'manual' ? '#c17a06' : '#208151';
        const x = (b[0] - view[0]) * scale, y = (b[1] - view[1]) * scale;
        const w = (b[2] - b[0] + 1) * scale, h = (b[3] - b[1] + 1) * scale;
        ctx.strokeStyle = color; ctx.lineWidth = chosen ? 3 : 2;
        ctx.setLineDash(entry.source === 'manual' ? [5, 3] : []);
        ctx.strokeRect(x, y, w, h); ctx.setLineDash([]);
        const label = String(index + 1); ctx.font = 'bold 13px Arial';
        const tw = ctx.measureText(label).width + 8, tx = Math.min(canvas.width - tw, x), ty = y >= 20 ? y - 19 : Math.min(canvas.height - 19, y + h + 2);
        ctx.fillStyle = color; ctx.fillRect(Math.max(0, tx), ty, tw, 18);
        ctx.fillStyle = '#fff'; ctx.fillText(label, Math.max(0, tx) + 4, ty + 13);
        if (chosen) {
          const c = entryCenter(entry), cx = (c[0] - view[0] + 0.5) * scale, cy = (c[1] - view[1] + 0.5) * scale;
          ctx.strokeStyle = color; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(cx - 5, cy); ctx.lineTo(cx + 5, cy); ctx.moveTo(cx, cy - 5); ctx.lineTo(cx, cy + 5); ctx.stroke();
        }
      });
    }
    function renderList() {
      list.replaceChildren();
      if (!state.entries.length) { list.append(element('p', 'lr-muted', 'No swatches found. Use Add by center to mark a visible legend symbol.')); return; }
      state.entries.forEach((entry, index) => {
        const item = button('', () => { if (busy()) return; state.selected = entry.id; state.mode = 'select'; render(); });
        item.setAttribute('aria-pressed', String(entry.id === state.selected));
        item.setAttribute('aria-label', 'Select swatch ' + (index + 1) + (entry.label ? ': ' + entry.label : ''));
        const thumbnail = element('canvas'); thumbnail.width = 112; thumbnail.height = 76;
        const ctx = thumbnail.getContext('2d'), b = entry.box, w = b[2] - b[0] + 1, h = b[3] - b[1] + 1;
        const ratio = Math.min(104 / w, 68 / h, 3);
        ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, 112, 76); ctx.imageSmoothingEnabled = false;
        ctx.drawImage(state.picture, b[0], b[1], w, h, (112 - w * ratio) / 2, (76 - h * ratio) / 2, w * ratio, h * ratio);
        const failed = unavailable(entry);
        const detail = String(index + 1) + (entry.label ? ' · ' + entry.label : '') +
          (failed ? ' (extraction failed)' : entry.source === 'manual' ? ' (edited)' : '') +
          (failed ? '\n' + (entry.reason || 'No usable marker template was extracted.') : '');
        item.append(thumbnail, element('span', failed ? 'lr-entry-failed' : '', detail));
        list.append(item);
      });
    }
    function deleteSelected() {
      if (busy() || !selectedEntry()) return;
      snapshot(); state.entries = state.entries.filter(entry => entry.id !== state.selected); state.selected = null; state.mode = 'select';
      render(); message(state.entries.length ? 'Swatch deleted. Apply the changes to rerun detection.' : 'All swatches removed. Add at least one correct swatch before rerunning.');
    }
    function undoEdit() {
      if (busy() || !state.undo.length) return;
      const previous = state.undo.pop(); state.entries = previous.entries; state.selected = previous.selected; render(); message('Last legend edit undone.');
    }
    function resetEdits() {
      if (busy()) return;
      snapshot(); state.entries = clone(state.initial); state.selected = null; state.mode = 'select'; render(); message('Restored the swatches from this detection result.');
    }
    function resizeSelected() {
      const entry = selectedEntry();
      if (busy() || !entry) return;
      const w = Number(width.value), h = Number(height.value);
      if (!Number.isInteger(w) || !Number.isInteger(h) || w < 3 || h < 3) { message('Enter whole numbers of at least 3 pixels for crop width and height.', true); return; }
      const center = entryCenter(entry);
      const box = resizedBox([...center, ...center], w, h, state.review.legend_box);
      if (JSON.stringify(box) === JSON.stringify(entry.box)) return;
      snapshot(); entry.box = box; entry.center = center; entry.source = 'manual'; entry.status = 'crop_edited'; delete entry.reason; delete entry.reason_code;
      render(); message('Crop resized. Check the source symbol in the preview, then apply the changes.');
    }
    canvas.addEventListener('click', async event => {
      if (busy() || !state.loaded) return;
      const point = canvasPoint(event.clientX, event.clientY, canvas.getBoundingClientRect(), state.view);
      if (!contains(state.review.legend_box, ...point)) { message('Click inside the selected legend region.', true); return; }
      if (state.mode === 'select') {
        // Prefer the smallest enclosing box when neighboring crops overlap.
        const matches = state.entries.filter(entry => contains(entry.box, ...point)).sort((a, b) =>
          (a.box[2] - a.box[0] + 1) * (a.box[3] - a.box[1] + 1) - (b.box[2] - b.box[0] + 1) * (b.box[3] - b.box[1] + 1));
        state.selected = matches[0]?.id || null; render();
        message(matches.length ? 'Selected swatch ' + (state.entries.indexOf(matches[0]) + 1) + '.' : 'No swatch at this position. Choose Add by center to add one.'); return;
      }
      if (state.mode === 'move') {
        const entry = selectedEntry(); if (!entry) return;
        snapshot(); const b = entry.box, w = b[2] - b[0] + 1, h = b[3] - b[1] + 1;
        entry.box = resizedBox([point[0], point[1], point[0], point[1]], w, h, state.review.legend_box);
        entry.center = point; entry.source = 'manual'; entry.status = 'crop_edited'; delete entry.reason; delete entry.reason_code;
        state.mode = 'select'; render(); message('Swatch center moved. Check its crop before applying.'); return;
      }
      state.pending = true; syncBusy(); message('Inspecting the source pixels around the selected center…');
      try {
        const payload = {center: point, entries: clone(state.entries), image_sha256: state.review.image.sha256};
        const answer = options.onSuggest ? await options.onSuggest(payload) : await readJSON('/legend-review/' + encodeURIComponent(data.job_id) + '/suggest', {
          method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)});
        if (!current()) return;
        const entry = clone(answer.entry || answer);
        if (!Array.isArray(entry.box) || entry.box.length !== 4 || !entry.box.every(Number.isFinite)) throw new Error('The suggested crop is invalid. Try another center.');
        if (!entry.id || state.entries.some(existing => existing.id === entry.id)) entry.id = 'manual_' + Date.now().toString(36);
        entry.source = 'manual'; entry.center = entryCenter(entry);
        snapshot(); state.entries.push(entry); state.selected = entry.id;
        const note = answer.warning || entry.proposal_note;
        render(); message('Swatch added. Check the whole symbol is inside its crop; resize if needed.' + (note ? '\n' + note : ''));
      } catch (error) { if (current()) message(error.message, true); }
      finally { if (current()) { state.pending = false; syncBusy(); } }
    });
    canvas.addEventListener('keydown', event => {
      if (event.key === 'Delete' || event.key === 'Backspace') { event.preventDefault(); deleteSelected(); }
      if (event.key === 'Escape') setMode('select');
    });
    async function rerun() {
      if (busy() || !changed() || !state.entries.length) return;
      if (!options.onRerun) { message('Rerunning legend edits is unavailable in this view.', true); return; }
      state.pending = true; syncBusy(); message('Rerunning detection with the reviewed legend swatches…');
      try {
        const result = await options.onRerun({job_id: data.job_id, entries: clone(state.entries), image_sha256: state.review.image.sha256});
        if (current() && !result) message('Detection was not rerun. Your legend edits are still available to apply.');
      } catch (error) { if (current()) message(error.message, true); }
      finally { if (current()) { state.pending = false; syncBusy(); } }
    }
    async function load() {
      if (!current()) return;
      if (state.loaded) { syncBusy(); return; }
      if (state.loading) return state.loading;
      state.loading = (async () => {
        message('Loading the original legend and extracted swatches…');
        try {
          const review = data.legend_review || await readJSON(data.legend_review_url || '/legend-review/' + encodeURIComponent(data.job_id));
          if (!current()) return;
          if (!review.legend_box || !review.image?.sha256) throw new Error('No reviewable legend region is saved for this result.');
          const picture = new Image();
          await new Promise((resolve, reject) => { picture.onload = resolve; picture.onerror = () => reject(new Error('The original legend image could not be loaded.')); picture.src = review.image_url || data.input_url || '/result/' + encodeURIComponent(data.job_id) + '/input.png'; });
          if (!current()) return;
          if (picture.naturalWidth !== review.image.width || picture.naturalHeight !== review.image.height) throw new Error('The saved image dimensions do not match the legend review. Run detection again.');
          state.review = review; state.picture = picture; state.entries = clone(review.entries || []); state.initial = clone(state.entries);
          const b = review.legend_box, pad = 7;
          state.view = [Math.max(0, b[0] - pad), Math.max(0, b[1] - pad), Math.min(picture.width - 1, b[2] + pad), Math.min(picture.height - 1, b[3] + pad)];
          const vw = state.view[2] - state.view[0] + 1, vh = state.view[3] - state.view[1] + 1;
          const scale = Math.min(4, Math.max(1, 560 / vw), 420 / vh);
          canvas.width = Math.max(1, Math.round(vw * scale)); canvas.height = Math.max(1, Math.round(vh * scale));
          state.loaded = true; contents.hidden = false; render();
          const warnings = (review.warnings || []).filter(Boolean).join('\n');
          message((state.entries.length ? 'Check for missing or incorrect swatches before applying changes.' : 'Automatic extraction found no swatches. Click Add by center to mark a visible legend symbol.') + (warnings ? '\n' + warnings : ''));
        } catch (error) { if (current()) { message(error.message, true); state.loading = null; } }
      })();
      return state.loading;
    }
    function clear() { state.alive = false; host.replaceChildren(); }
    syncBusy();
    return {load, clear, syncBusy};
  }
  root.LegendReviewV46 = {mount};
  // Pure geometry helpers support coordinate-regression checks without a browser.
  if (typeof module !== 'undefined' && module.exports) module.exports = {resizedBox, canvasPoint, contains};
})(typeof window !== 'undefined' ? window : globalThis);
