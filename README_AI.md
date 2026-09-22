# Plot Digitizer v46 — AI maintenance guide

This repository is a standalone runtime extracted from a larger research workspace.
The runtime sources are copied without changing their algorithms. Training data,
experiments, evaluation outputs, obsolete trained ViT entry points and model
weights are intentionally absent. Do not recreate dependencies on the original
workspace or assume its absolute paths, `experiments/` or `models/` are available.

## Start and dependencies

Use Python 3.12 and `python -m pip install -r requirements.txt`, then `python run.py`.
The launcher resolves `src` relative to itself and binds to `127.0.0.1:8000` by
default. `--port` selects another port; use an unused port for tests and stop only
the server you started. Direct `src/unified_server.py` execution retains its
original `0.0.0.0:8000` behavior. No Node build step is needed for the browser UI.

CPU execution does not require PyTorch, timm, YOLO or trained model weights.
`bw_compute_backend.py` lazily probes optional PyTorch/CUDA; auto selection falls
back to CPU, whereas an explicitly requested CUDA backend is strict. Set
`BW_V46_REFINEMENT_BACKEND=cpu` for deterministic CPU smoke checks. OCR uses the
`pytesseract` wrapper and an optional external Tesseract executable on PATH.
Do not make OCR or CUDA a prerequisite for importing the web server.

## Operating this app for a user from Codex or Claude Code

When the user asks you to run digitization, carry out setup and launch the app;
do not stop at explaining installation commands. Use the repository URL and
destination supplied by the user. Clone/download that repository into the local
workspace if it is not already available; otherwise use the existing checkout
and preserve its edits. Do not invent a GitHub URL or require the research repo.

Read this guide and `README.md`, create a local Python 3.12 virtual environment,
and install `requirements.txt` using that environment's Python. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py --ai-ocr --port 8000
```

On macOS/Linux use `python3.12 -m venv .venv`,
`.venv/bin/python -m pip install -r requirements.txt`, and
`.venv/bin/python run.py --ai-ocr --port 8000`. Installation was verified on Windows;
resolve platform-specific installation failures before claiming readiness.
If the port is occupied, choose another available port rather than stopping an
unrelated process. Keep the server alive while the user interacts, verify
`/openapi.json` using local HTTP tools, and give the exact local URL. Use the
agent's supported background process mechanism; Windows background helpers
should start without opening an unwanted terminal window.

### Tesseract unavailable: operate the real AI OCR bridge

**Start `python run.py --ai-ocr` when operating with image-viewing tools.** Read
[src/AI_OCR.md](src/AI_OCR.md) before running detection. The v46 runtime now
exports its actual OCR snippets when Tesseract is missing or fails. View those
PNGs and submit text or word boxes; the waiting detector consumes your answer
and continues the same operation. Do not merely describe the text in chat.

Keep the long-running server/detection request in the background while you poll
`GET /ocr/requests`, inspect each PNG, and POST validated JSON to its `answer_url`.
The guide gives exact schemas and equivalent file-only commands. Sequential
requests can appear after earlier ones are answered; keep servicing the queue
until detection/correction itself completes. A temporary empty queue is not a
completion signal. Keep each request bound to its job, source image and hash.

Tesseract remains first choice. Importing `pytesseract` does not prove that the
executable works; the bridge checks it and catches OCR execution failures.
The bridge does not judge whether successfully returned Tesseract text is
correct. `CHARTOCODE_OCR_MODE=agent` explicitly forces agent OCR when needed.
No API key, paid vision service or second assistant process is required.
Python cannot itself call the current conversation: **the operating agent must
actively answer the queue**. A README instruction alone does not run the agent.

Use only clearly visible text. Preserve signs, decimals, exponents, units and
legend-to-swatch identities. Word boxes use snippet pixels, not chart pixels.
For unreadable snippets submit `status: "unreadable"`, then ask for necessary
missing values; never invent numbers or submit empty answers without inspection.
Missing answers time out explicitly after 300 seconds by default. If the current
agent cannot view images, use ordinary mode with Tesseract or manual axis values.

Review AI-read labels and calibration with the user. Manual axis ranges must
match the selected plot edges. Use the optional **scale box** when readable ticks
are inside the plot, so the UI converts their ranges to the plot boundaries.
Direct HTTP calls must use converted plot-edge ranges. Retain series IDs,
coordinates and evidence, and visibly identify unknown/pixel-unit calibration.

### Let the user draw plot and legend regions

The existing `src/index.html` already provides user-controlled rectangles.
Open the running page, let the user choose the image, and ask them to press
**Plot area** and drag over the plotting region, then **Legend** and drag over
the legend if one exists. Use the application's buttons/canvas, not browser
annotation mode. Let the user retain control when they asked to select regions.
After image preparation or changing the image, have them reselect the regions.

Choose the browser handoff according to the actual agent environment:

| Environment | User-facing route |
| --- | --- |
| Codex in a desktop app with the built-in browser available | Open the local URL visibly in that browser (for example via `open_in_codex` when that tool is exposed). The user selects the image and draws on the page. |
| Codex CLI / an IDE without a built-in browser | Give the local URL for the user's normal Chrome/Edge browser. Browser automation requires an available browser integration; web search/fetch alone is insufficient. |
| Claude Code with browser integration | Use its connected Chrome/Chromium browser; `claude --chrome` / `/chrome` are documented entry points when supported and configured. The user can interact with the visible page. |
| Claude Code without browser integration | Give the same local URL for the user to open manually. The app does not require an agent browser extension for human use. |

Codex desktop browser operation and local-page previews are described in the
[official browser documentation](https://learn.chatgpt.com/docs/browser?surface=app).
Claude Code's setup requirements and local-server interaction are described in
[Claude Code with Chrome](https://code.claude.com/docs/en/chrome). Check the tools
available in the current session rather than assuming every CLI has a webview.
Let the user operate file pickers when the agent cannot automate uploads.

**WebFetch is a content-retrieval tool, not an interactive browser.** It does not
give the user a canvas, process drag gestures or substitute for JavaScript page
execution. Anthropic documents this distinction in its
[web fetch guide](https://platform.claude.com/docs/en/agents-and-tools/tool-use/web-fetch-tool).
Fetching `index.html` or `/docs` is not a completed region-selection interaction.
Cloud fetchers may also be unable to reach the user's localhost. If the server
runs remotely, use the environment's supported port forwarding; `localhost`
refers to the machine running the browser, not automatically the remote agent.

### How selected regions reach detection and the AI

`index.html` stores the rectangles in browser state (`boxes.plot` and
`boxes.legend`). The user presses **Run** after choosing regions and reviewing
axis values. The UI then posts the image and coordinates to `/digitize`.
For Colour the legend field is `legend_box`; for B&W it is `legend_area`.
All web rectangles use inclusive `x0,y0,x1,y1` working-image pixels.
The server validates them against the uploaded raster.

Selecting a rectangle alone does not send it to the AI conversation or persist
it through an API. There is currently no dedicated pending-selection endpoint
or browser-to-chat callback. Do not claim WebFetch can read unsent rectangles.
After **Run**, use the returned job/result URLs through available browser tools,
or let the user save the image-linked JSON and provide its path. Resume from that
identified result, rather than guessing which unrelated temporary job is newest.
If you need rectangles before **Run**, use the connected browser's supported
state-inspection capability or ask the user for the coordinates.

A missing legend means the current no-legend route infers **one series**; it is
not permission to assume an arbitrary number of hidden legend entries. Plot
selection remains required for this route. Show the overlay/reconstruction for
review and preserve the original image together with the saved JSON.

## Runtime map

| Component | Responsibility and connections |
| --- | --- |
| `src/ocr_bridge_v46.py` | Tesseract adapter and local AI snippet/answer queue; protocol in `src/AI_OCR.md` |
| `src/unified_server.py` | FastAPI routes, AI OCR request/image/answer endpoints, image preparation, detector/correction child processes, job responses, saved-session import/export |
| `src/index.html` | HTML/CSS and inline JavaScript; image preparation, result history, folder previews, fixed editor, canvas drawing, exports |
| `src/image_sidecar_v46.js` | Browser folder/file handles, original-image binding, same-stem JSON pairing, history merge and writes |
| `src/run_A4_auto_v46.py` | Shared v46 dispatcher: Colour → `run_A4_color_v46.py`; B&W → `bw_detect_cli.py` with `grid_v46` |
| `src/run_A4_color_v46.py` | Routes optional-legend cases, calls the standalone colour workflow and optional Step 5 |
| `src/color_pipeline_v46.py` | Explicit image/palette preparation, v46 detection, calibration and result export |
| `src/chart_analysis_v46.py` | Import-safe shared pixel, colour, legend, calibration and rendering functions |
| `src/legend_palette_v46.py` | Named palette filtering, recovery, locking and swatch-selection stages |
| `src/analysis_session_v46.py` | Per-image analysis namespace and isolated stage-function bindings |
| `src/color_legend_runtime_v46.py` | Legend composition, explicit prepared colour samples and template adapters |
| `src/color_marker_runtime_v46.py` | Current colour marker extraction and routing, including same-colour groups |
| `src/bw_pipeline.py`, `src/bw_pipeline_v46.py` | Shared B&W orchestration and model-free legend/grid detector |
| `src/legend_optional_v46/` | No-legend single-series inference and automatic legend/source routing |
| `src/type3_v46/` | Marker-free line/error-bar evidence, endpoints, reconstruction/correction adapters |
| `src/color_correct_cli.py`, `src/color_path_correction_v46.py` | Colour Step-5 path/structure correction using saved detection evidence |
| `src/bw_step5_v46.py`, `src/bw_series_correction_v46.py` | B&W Step-5 correction and structure-dependent scoring |
| `src/correction_session_v46.py` | Session validation, image/evidence binding, import/export, Excel, saved-only correction |
| `src/correction_history_v46.py` | Version graph, original/prepared image objects, deduplicated evidence and history export |
| `src/correction_only_cli_v46.py` | Runs correction from saved evidence; must not rerun detection |

The many `color_*`, `bw_*`, legend and geometry helpers are transitive runtime
dependencies. `RUNTIME_MANIFEST.json` lists the exact packaged source files,
their initial SHA-256 hashes and dependency reference locations. Some references
are string-based or diagnostic-only, so the manifest is provenance, not a full
call graph or a promise that every conditional branch is supported.

## Standalone v46 analysis and loading

`run_A4_auto_v45.py` is absent and must remain unnecessary. Do not restore legacy
source reads, AST transforms, executable-source hashes or line-range extraction.
`run_A4_auto_v46.py` dispatches scripts with `runpy.run_path`; the colour wrapper
then calls `color_pipeline_v46.run()` as a normal Python function.

`AnalysisSession` loads the import-safe `chart_analysis_v46` library into a fresh
module namespace for each image. Many numerical helpers use that namespace for
image arrays and palette state. `install()` binds a stage function's existing
code to the same namespace using `FunctionType`; stage code is not generated or
rewritten. Keep image state private: do not import one shared mutable namespace
for every request. `legend_palette_v46` stages are used by both the colour
workflow and `legend_optional_v46.native_legend.extract()`.

`LegendRuntime.colour_masks()` supplies observed samples through the explicit
`samples=` argument, bypassing broad swatch re-sampling without inspecting the
helper's source. Calibration guards and export hooks are direct workflow calls.
Missing marker evidence raises a diagnostic instead of selecting an older
colour/walk detector. Auto/line-only routes remain supported. The obsolete
`--no-color-pipeline` option and `compile_v45_bridge()` API are removed.

The server selects `run_A4_auto_v46.py` explicitly. A `PLOT_PIPELINE` override can
point to another copy of that entry point; a legacy filename is rejected.
Numbered Step-5 files (`3_segment_detection_v2.py`, `4_segment_refinement.py`,
`5_correction_color.py`) are still loaded by filename. Keep those runtime files.
Legacy ViT-only compatibility paths are not distributed or used by the web
v46 dispatcher. CPU operation needs no model weights.

## Invariants to preserve

### Coordinates and evidence

The web editor edits points in working-image pixel coordinates. Axis calibration
converts pixels to values without changing the source points. User-submitted web
ROIs use inclusive right/bottom bounds within `width-1`, `height-1`; several
internal v46 geometry helpers use half-open bounds. Convert at existing boundaries
rather than changing one convention globally.

Resolution/rotation creates a new working image and invalidates incompatible
selections and results. Original image binding and prepared-image evidence are
separate. Match source fingerprints, dimensions, series identities and calibration
when importing; never accept an unrelated JSON just because the filename matches.

Suppressed candidates and inferred hypotheses are evidence, not automatically
observed measurements. Preserve their provenance and serialized state. Missing or
uncertain detection evidence must not silently invoke an obsolete detector.

### Save contract

`chartocode-v46-image-history-v2` is the versioned image-linked format. Older
image-sidecar/session formats remain readable. Preserve initial detection,
Step-5 branches, manual edits and deduplicated evidence. Saving should not mutate
an earlier server job or delete a previously saved branch.

`saveCorrection()` sends `edit_data` as an `application/json` Blob with a filename.
Keep that multipart **file** contract: large multipart text fields hit Starlette's
separate 1MiB limit. The server accepts `UploadFile | str` for compatibility and
enforces the application JSON limit of 128MiB with bounded reads.
`checkedResponse()` / `responseErrorMessage()` must show structured errors as
readable messages, never `[object Object]`.

`currentEdForSave()` ensures the edits belong to the current job. Save/export uses
all curves even when the UI filters to one colour. Correct-saved processing forks
the saved job and consumes saved evidence without detection.

### Detection time and failures

`DETECTION_TIMEOUT_SECONDS` is 900 for ordinary Colour and B&W requests. AI-assisted OCR retains its separate overall allowance. `/digitize` catches child-process timeouts and returns `ok: false`, `timed_out: true` and a readable summary; valid partial series may still be manually edited, but Step 5 stays disabled. `_response()` persists complete UTF-8 output in `pipeline.log` and returns `log_url`; keep the short inline log too. The Run elapsed-time timer must stop on success, failure and stale responses.

CPU and CUDA candidate-centre selection share `_suppress_ranked_centres()`. Spatial buckets only prune distant comparisons: stable rank order, inclusive distance boundaries, point coordinates and vote-map arithmetic must remain identical to the original greedy rule.

### Editor contract

`ok` describes automatic detection/correction success; `editable_available` separately
reports whether the saved image and series data pass validation for manual editing.
Keep Edit, Step 5 and legend-analysis controls visible after Run. Do not hide all
actions when inference is uncertain, and do not enable Step 5 merely because manual
edits exist. `resultCapabilities()` also guards controls after a busy operation ends.
Unresolved but editable jobs retain session metadata for JSON/history/Excel export.
The legend panel can show `legend_optional` and its JSON link without a PNG diagnostic.


- `initEditor()` guards asynchronous loads so stale responses cannot revive an
  old image. `openEditorView()` switches to a fixed full-window editor;
  `closeEditorView()` restores the previous workspace without saving or discarding
  in-memory edits. Resume and reopening the current job preserve edits.
- `edTool` selects Edit points or Pan. An empty-space press is deferred until
  pointer release: a click adds a point, while movement beyond the gesture
  threshold pans. Point drags edit coordinates. Pan can start on points without
  moving them. Pointer cancellation/capture loss must clear gesture state.
- `editSurface` provides scroll extent. `editCanvas` allocates only the visible
  viewport; `edViewX`/`edViewY` translate drawing and hit testing. Do not restore
  `max-width:100%` on this canvas or allocate a bitmap the size of the entire
  zoomed image. Zoom changes view state, not stored points.
- `buildCurvePicker()` updates both the editable overlay and reconstruction.
  `drawRecon()` filters displayed curves while using all points for stable axes;
  all-curve data remain available for saves and the coordinate table.
- `drawRecon()` is a display-only x-ordered straight-line view of current points.
  Do not overwrite the saved Step-5 reference or fit state with this visualization.
- Folder preview navigation and the editor's loaded file are independent.
  Unsaved-change and busy guards must survive switching adjacent files.

## Verification when changing code

1. Compile changed Python modules and import the server in a clean environment.
   Confirm no v45 Python files are present or required. Every shipped Python module is import-safe.
2. Start `run.py` on an unused loopback port from a working directory outside this
   repository. Confirm `/`, `/image_sidecar_v46.js` and `/openapi.json` respond.
3. Use generated charts for Colour markers, B&W markers and a no-legend/line-only
   case. Supply explicit calibration so OCR is not a test prerequisite. Inspect
   output points/overlays; synthetic smoke checks do not establish real-chart accuracy.
4. Exercise supported Step-5 correction and saved-only correction. Confirm the
   detection stage is not re-entered on import or saved-only correction.
5. Round-trip image + JSON, manual edits, version history and Excel; include an
   `edit_data` multipart file larger than 1MiB and verify all points survive.
6. For UI changes, verify folder previews/arrows, opening/closing/resuming the
   editor, one-colour reconstruction, point drag/add/delete, background/Pan drag,
   high zoom, Fit, viewport resizing and save buttons visible on screen.

The original research test corpus is intentionally not included. Keep future
tests small and self-contained; do not add private user charts or absolute paths.
Regenerate the manifest hashes when runtime code changes. The standalone migration
was checked with 143 runtime files, 141 imports, five detection/correction cases,
exact point/label/colour/calibration parity, and a 55,000-point JSON/Excel round trip.

## Communication

Explain the current behavior, specific problem, change and reason in plain
language. Name exact files/functions and report what was actually tested, including
limitations. Preserve unrelated edits. Do not publish, push, change licensing, or
create a remote repository unless the user asks for that action.

Focused migration regressions: 83 passed, including ten real legend fixtures.
The broader optional-legend suite passes 111/113; the two failures still expect
`bw_step5_v46` instead of the existing `bw_series_correction_v46` engine name.
Both failures were reproduced on the untouched pre-migration runtime.

AI OCR bridge verification: missing-Tesseract fallback, native Tesseract preference,
word-box validation, image binding, timeout behavior and live HTTP answer handling
are covered by focused tests. Agent OCR is opt-in; ordinary runs do not wait for an agent.
The old path-server test still expects v45 dispatch compatibility; that failure
was reproduced on the pre-OCR distribution and does not describe the v46-only contract.
