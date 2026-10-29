/* Chat selects the workflow; AI mode selects files without replacing open results. */
(function () {
  'use strict';
  const API = '/agent-workflow';
  const $ = id => document.getElementById(id);
  const params = new URLSearchParams(location.search);
  const batchId = params.get('batch'), itemId = params.get('item');
  const state = {agentMode: false, available: false, agentActive: false,
    selected: [], skipped: 0, uploading: false, refreshing: false, timer: null,
    batches: new Map(), resultNotice: '', statusFailures: 0};
  const storageKey = 'digitizer-ai-tab-batches';
  const originalSubtitle = document.querySelector('#mainWorkspace > .sub').textContent;
  const pageSize = 50;

  async function request(path, options = {}) {
    const response = await checkedResponse(await fetch(path, {cache: 'no-store', ...options}));
    return response.json();
  }

  function selectionError() {
    if (state.selected.length > 1000) return 'Select no more than 1,000 images per batch.';
    const large = state.selected.find(file => file.size > 100 * 1024 * 1024);
    if (large) return (large.webkitRelativePath || large.name) + ' exceeds 100 MiB. Select a smaller image.';
    if (state.selected.reduce((total, file) => total + file.size, 0) > 2 * 1024 * 1024 * 1024)
      return 'The selected images exceed 2 GiB. Select a smaller batch.';
    return '';
  }

  function controls() {
    $('aiChooseImages').disabled = state.uploading;
    $('aiChooseFolder').disabled = state.uploading;
    $('aiImagesInput').disabled = state.uploading;
    $('aiFolderInput').disabled = state.uploading;
    $('aiStartDigitization').disabled = state.uploading || !state.selected.length || !state.available || !!selectionError();
    $('aiClearSelection').disabled = state.uploading || !state.selected.length;
  }

  function applyStatus(value) {
    state.available = true;
    state.statusFailures = 0;
    state.agentActive = value.agent?.active === true;
    state.agentMode = value.settings?.region_mode === 'agent';
    document.body.classList.toggle('agent-input-mode', state.agentMode);
    $('aiFilePicker').hidden = !state.agentMode;
    document.querySelector('#mainWorkspace > .sub').textContent = state.agentMode
      ? 'Select images for your AI agent, then review and correct the recovered points.' : originalSubtitle;
    $('aiAgentStatus').textContent = state.agentActive
      ? 'AI agent connected. Keep the chat agent running while your images are processed.'
      : 'Waiting for your AI agent. You can queue images now; processing waits for the agent to select regions and read text.';
    if (state.agentMode && state.resultNotice) $('aiSubmissionStatus').textContent = state.resultNotice;
    controls();
  }

  function chooseFiles(files) {
    if (state.uploading) return;
    const all = Array.from(files);
    if (!all.length) return; // Cancelling a picker preserves the current selection.
    state.selected = all.filter(file => /\.(png|jpe?g)$/i.test(file.name)).sort((a, b) =>
      (a.webkitRelativePath || a.name).localeCompare(b.webkitRelativePath || b.name, undefined, {numeric: true}));
    state.skipped = all.length - state.selected.length;
    state.resultNotice = ''; $('aiSubmissionStatus').textContent = '';
    renderSelection();
  }

  function renderSelection() {
    const count = state.selected.length;
    $('aiSelectionCount').textContent = (count ? count + ' image' + (count === 1 ? '' : 's') + ' selected.' : 'No images selected.') +
      (state.skipped ? ' ' + state.skipped + ' unsupported file' + (state.skipped === 1 ? '' : 's') + ' excluded; PNG/JPEG only.' : '') +
      (count > 100 ? ' Showing the first 100 names.' : '') +
      (selectionError() ? ' ' + selectionError() : count ? ' Start digitization uploads all selected images.' : '');
    const list = $('aiSelectedFiles'); list.replaceChildren(); list.hidden = !count;
    for (const file of state.selected.slice(0, 100)) {
      const row = document.createElement('li'); row.textContent = file.webkitRelativePath || file.name; list.appendChild(row);
    }
    controls();
  }

  function rememberBatches() {
    try { sessionStorage.setItem(storageKey, JSON.stringify([...state.batches.keys()])); } catch (_) {}
  }

  function itemStatus(item) {
    if (item.waiting_ocr || item.pending_ocr_count) return 'Waiting for OCR';
    return ({pending_regions: 'Waiting for regions', ready: 'Queued', waiting_agent: 'Waiting for agent',
      waiting_ocr: 'Waiting for OCR', running: 'Running', completed: 'Completed — review required',
      failed: 'Failed', needs_review: 'Needs review', cancelled: 'Cancelled'})[item.status] || item.status || 'Queued';
  }

  function renderBatches() {
    const wrap = $('aiBatchResults'); wrap.hidden = !state.batches.size;
    for (const [id, entry] of state.batches) {
      const batch = entry.value;
      let section = entry.section;
      if (!section) {
        section = document.createElement('section'); section.className = 'ai-batch';
        entry.section = section; wrap.prepend(section);
      }
      const items = batch?.items || [];
      entry.page = Math.min(entry.page || 0, Math.max(0, Math.ceil(items.length / pageSize) - 1));
      const signature = JSON.stringify([batch, entry.page, entry.error]);
      if (signature === entry.signature) continue;
      entry.signature = signature; section.replaceChildren();
      const heading = document.createElement('h3'); heading.textContent = 'Batch ' + id; section.appendChild(heading);
      const summary = document.createElement('p'); summary.className = 'hint';
      summary.textContent = entry.error || (batch
        ? items.filter(item => item.status === 'completed').length + ' / ' + items.length + ' completed · ' + String(batch.status || 'queued').replaceAll('_', ' ')
        : 'Loading batch progress…');
      section.appendChild(summary);
      if (!batch) continue;
      const tableWrap = document.createElement('div'); tableWrap.className = 'ai-table-wrap';
      const table = document.createElement('table'), head = document.createElement('thead'), header = document.createElement('tr');
      for (const label of ['Image', 'Status', 'Result']) {
        const cell = document.createElement('th'); cell.scope = 'col'; cell.textContent = label; header.appendChild(cell);
      }
      head.appendChild(header); table.appendChild(head);
      const body = document.createElement('tbody');
      for (const item of items.slice(entry.page * pageSize, (entry.page + 1) * pageSize)) {
        const row = document.createElement('tr'), name = document.createElement('td'), status = document.createElement('td'), action = document.createElement('td');
        name.textContent = item.name; status.textContent = itemStatus(item);
        if (item.error) {
          const error = document.createElement('p'); error.className = 'hint'; error.textContent = responseErrorMessage(item.error); status.appendChild(error);
        }
        if (item.review_error) {
          const error = document.createElement('p'); error.className = 'hint';
          error.textContent = 'Saved review files: ' + responseErrorMessage(item.review_error); status.appendChild(error);
        }
        if (item.result?.input_url && item.result?.edit_data_url) {
          const link = document.createElement('a'); link.textContent = 'Review';
          link.href = '/?batch=' + encodeURIComponent(id) + '&item=' + encodeURIComponent(item.item_id);
          link.target = '_blank'; link.rel = 'noopener'; action.appendChild(link);
        }
        row.append(name, status, action); body.appendChild(row);
      }
      table.appendChild(body); tableWrap.appendChild(table); section.appendChild(tableWrap);
      if (items.length > pageSize) {
        const nav = document.createElement('div'); nav.className = 'ai-actions';
        const previous = document.createElement('button'), next = document.createElement('button'), count = document.createElement('span');
        previous.type = next.type = 'button'; previous.textContent = 'Previous'; next.textContent = 'Next';
        previous.disabled = entry.page === 0; next.disabled = (entry.page + 1) * pageSize >= items.length;
        previous.onclick = () => { entry.page--; renderBatches(); }; next.onclick = () => { entry.page++; renderBatches(); };
        count.textContent = (entry.page * pageSize + 1) + '–' + Math.min((entry.page + 1) * pageSize, items.length) + ' / ' + items.length;
        nav.append(previous, count, next); section.appendChild(nav);
      }
      if (items.some(item => item.result)) {
        const download = document.createElement('a'); download.textContent = 'Download batch results (.zip)';
        download.href = batch.download_url || API + '/batches/' + encodeURIComponent(id) + '/results.zip';
        download.download = ''; section.appendChild(download);
      }
      if (batch.review_dir && items.some(item => item.review_json_path)) {
        const path = document.createElement('p'); path.className = 'hint';
        path.textContent = 'Saved image + JSON pairs: ' + batch.review_dir + '. Open this folder with Edit saved results below.';
        section.appendChild(path);
      }
    }
  }

  async function refresh() {
    if (state.refreshing) return;
    state.refreshing = true; clearTimeout(state.timer);
    try {
      applyStatus(await request(API + '/status'));
      if (state.agentMode) {
        await Promise.all([...state.batches].map(async ([id, entry]) => {
          try { entry.value = await request(API + '/batches/' + encodeURIComponent(id)); entry.error = ''; }
          catch (error) { entry.error = 'Could not refresh this batch: ' + error.message; }
        }));
        renderBatches();
      }
    } catch (_) {
      state.available = false;
      state.statusFailures++;
      $('aiAgentStatus').textContent = 'The local server could not be reached. New uploads are paused until it reconnects.';
      controls();
    } finally {
      state.refreshing = false;
      const interval = state.agentMode ? 4000 : 15000;
      const backoff = Math.pow(2, Math.min(Math.max(state.statusFailures - 1, 0), 3));
      state.timer = setTimeout(refresh, Math.min(30000, interval * backoff));
    }
  }

  async function startBatch() {
    if (state.uploading || !state.selected.length) return;
    const error = selectionError();
    if (error) { $('aiSubmissionStatus').textContent = error; return; }
    state.uploading = true; state.resultNotice = ''; controls();
    $('aiSubmissionStatus').textContent = 'Uploading selected images…';
    try {
      const status = await request(API + '/status'); applyStatus(status);
      if (!state.agentMode) throw new Error('The chat selected manual regions. No images were uploaded.');
      const form = new FormData();
      for (const file of state.selected) form.append('files', file, file.webkitRelativePath || file.name);
      form.append('options', JSON.stringify({region_mode: 'agent', ocr_mode: 'agent', mode: 'auto', correct: false, fast_detection: false}));
      const batch = await request(API + '/batches/upload', {method: 'POST', body: form});
      if (!batch.batch_id) throw new Error('The server did not return a batch identifier.');
      state.batches.set(batch.batch_id, {value: batch, page: 0}); rememberBatches(); renderBatches();
      state.selected = []; state.skipped = 0; $('aiImagesInput').value = ''; $('aiFolderInput').value = ''; renderSelection();
      $('aiSubmissionStatus').textContent = 'Images queued. Review links open in new tabs; your current edits stay in this tab.';
    } catch (error) {
      $('aiSubmissionStatus').textContent = 'Could not queue the selected images: ' + error.message;
    } finally {
      state.uploading = false; controls(); refresh();
    }
  }

  function resultStatus(text) {
    state.resultNotice = text; $('bwImageStatus').textContent = text;
    if (state.agentMode) $('aiSubmissionStatus').textContent = text;
  }

  async function getOcrMode() {
    try {
      const response = await fetch(API + '/settings', {
        cache: 'no-store', signal: AbortSignal.timeout(5000)
      });
      if (!response.ok) return '';
      const settings = await response.json();
      return ['tesseract', 'agent', 'auto-agent'].includes(settings.ocr_mode) ? settings.ocr_mode : '';
    } catch (_) {
      // Older servers keep their existing OCR default when this API is absent.
      return '';
    }
  }

  async function openExplicitResult(batchId, itemId) {
    // Only the initial page URL requests a result. Never replace work begun in
    // this tab or discover/open other results in the background.
    if (imageState.busy || fileBlob || window.lastData) {
      resultStatus('Open the result link in a new tab to keep the current image and edits.');
      return;
    }
    const token = imageState.begin('result-link');
    syncImageControls();
    resultStatus('Opening the saved result…');
    let sourceUrl = null, workingUrl = null, installed = false;
    try {
      const batch = await request(API + '/batches/' + encodeURIComponent(batchId));
      const item = batch.items?.find(value => value.item_id === itemId);
      if (!item) throw new Error('The requested image was not found in this batch.');
      const data = item.result;
      if (!data?.input_url || !data?.edit_data_url || !item.image_url) {
        throw new Error('This image does not yet have a saved editable result.');
      }
      const [saved, sourceResponse, workingResponse] = await Promise.all([
        request(data.edit_data_url),
        fetch(item.image_url).then(checkedResponse),
        fetch(data.input_url).then(checkedResponse)
      ]);
      if (!Array.isArray(saved.curves) || !Array.isArray(saved.plot_area) ||
          saved.plot_area.length !== 4 || !saved.plot_area.every(Number.isFinite)) {
        throw new Error('The saved result has no valid editable plot coordinates.');
      }
      const [sourceBlob, workingBlob] = await Promise.all([sourceResponse.blob(), workingResponse.blob()]);
      const name = (item.name || 'plot.png').split(/[/\\]/).at(-1);
      const source = new File([sourceBlob], name, {type: sourceBlob.type || 'image/png'});
      const digest = await ImageSidecarV46.digest(source);
      if (item.image_sha256 && digest !== item.image_sha256) {
        throw new Error('The original image does not match this saved result.');
      }
      sourceUrl = URL.createObjectURL(source);
      workingUrl = URL.createObjectURL(workingBlob);
      const [original, working] = await Promise.all([decodeImage(sourceUrl), decodeImage(workingUrl)]);
      if (saved.image && (saved.image.width !== working.width || saved.image.height !== working.height)) {
        throw new Error('The saved coordinates and working image have different dimensions.');
      }
      if (!imageState.current(token)) return;

      // Commit only after the original, working image and editable JSON load.
      // The original File remains the source for existing linked-JSON exports.
      imageState.finish(token);
      clearImageGeometry(); revokePreparedUrl();
      if (originalUrl) URL.revokeObjectURL(originalUrl);
      originalUrl = sourceUrl; preparedUrl = workingUrl; originalImage = original;
      const upload = imageState.upload(source);
      imageState.commitUpload(upload, {width: original.width, height: original.height});
      imageState.currentBlob = workingBlob;
      imageState.currentSize = {width: working.width, height: working.height};
      imageState.finish(upload);
      installed = true;
      sidecar.source = source; sidecar.sourceDigest = digest; sidecar.directory = null;
      folderEditorSource = null; folderEditorDirectory = null; savedFolderState = null;
      boxes.plot = saved.plot_area.slice();
      const regions = data.fast_detection?.working_regions;
      boxes.legend = regions?.legend_box || regions?.legend_area ||
        (!data.fast_detection?.applied ? item.regions?.legend_box || null : null);
      document.getElementById('useScaleBox').checked = false;
      scaleBoxUpdate();
      for (const axis of ['x', 'y']) {
        const calibration = saved.calibration?.[axis];
        const manual = calibration && !['pixel', 'normalized'].includes(calibration.kind) &&
          calibration.status !== 'needs_calibration' &&
          !['automatic', 'mixed', 'pixel', 'uncalibrated_pixel_distance'].includes(calibration.source);
        document.getElementById(axis + 'Min').value = manual ? calibration.v0 : '';
        document.getElementById(axis + 'Max').value = manual ? calibration.v1 : '';
        document.getElementById(axis + 'Log').checked = !!(manual && calibration.log);
      }
      document.getElementById('bwResolution').value = '100';
      showImage(working, workingBlob); updateInfo(); presentResult(data);
      resultStatus('Saved result loaded. Use Edit data points to review and correct the recovered points.');
    } catch (error) {
      resultStatus('Could not open the saved result: ' + error.message);
    } finally {
      if (!installed) {
        if (sourceUrl) URL.revokeObjectURL(sourceUrl);
        if (workingUrl) URL.revokeObjectURL(workingUrl);
      }
      imageState.finish(token); syncImageControls();
    }
  }

  window.AgentWorkflowV46 = {getOcrMode};
  try {
    const remembered = batchId && itemId ? [batchId] : JSON.parse(sessionStorage.getItem(storageKey) || '[]');
    if (Array.isArray(remembered)) for (const id of remembered) if (typeof id === 'string' && /^[A-Za-z0-9_-]+$/.test(id)) state.batches.set(id, {page: 0});
  } catch (_) {}
  $('aiChooseImages').onclick = () => { $('aiImagesInput').value = ''; $('aiImagesInput').click(); };
  $('aiChooseFolder').onclick = () => { $('aiFolderInput').value = ''; $('aiFolderInput').click(); };
  $('aiImagesInput').onchange = event => chooseFiles(event.target.files);
  $('aiFolderInput').onchange = event => chooseFiles(event.target.files);
  $('aiClearSelection').onclick = () => { state.selected = []; state.skipped = 0; renderSelection(); };
  $('aiStartDigitization').onclick = startBatch;
  window.addEventListener('focus', refresh);
  refresh();
  if (batchId && itemId) openExplicitResult(batchId, itemId);
})();

