# Plot Digitizer v46 — AI maintenance guide

This repository is a standalone runtime extracted from a larger research workspace.
The runtime sources are copied without changing their algorithms. Training data,
experiments, evaluation outputs, obsolete trained ViT entry points and model
weights are intentionally absent. Do not recreate dependencies on the original
workspace or assume its absolute paths, `experiments/` or `models/` are available.

## Start and dependencies

Use Python 3.12 and `python -m pip install -r requirements.txt`, then `python run.py`.
A plain launch selects manual GUI regions and Tesseract, independently of
stored workflow settings. It does not require an agent or a chat questionnaire.
Agent choices are supplied through CLI/API when an agent is operating the app.
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

Read [src/AI_WORKFLOW.md](src/AI_WORKFLOW.md) before starting a digitization task.
That guide defines startup choices, verified Tesseract setup, region and OCR
queues, folder scope, the active service loop, and required human review.

When the user asks you to run digitization, carry out setup and launch the app;
do not stop at explaining installation commands. Use the repository URL and
destination supplied by the user. Clone/download that repository into the local
workspace if it is not already available; otherwise use the existing checkout
and preserve its edits. Do not invent a GitHub URL or require the research repo.

Ask the region question first **in the conversation**, and ask about OCR only
for manual regions, skipping every answer
already supplied:

1. **Regions:** "Would you like to select the plot and legend regions yourself,
   or have the AI agent identify them?"

   Include this timing note with the question, before the user chooses:
   "AI selection reduces manual work but usually takes longer than selecting the regions yourself in the GUI."
   Offer **Select regions myself — usually faster** and
   **Let the AI agent select regions — less manual work, usually slower**.

2. Only for manual regions: "Should I use Tesseract, installing it if it is missing, or read the text
   with my AI vision tools?"

Ask the region question first. AI regions automatically use the operating
agent's vision OCR, without another question. Ask about Tesseract versus agent
OCR only if manual regions were chosen and the OCR answer is not already known.
Do not ask for paths. Apply the resulting choice through
CLI/API. In agent-region mode, open the AI-mode GUI input picker: **Choose images**
accepts multiple PNG/JPG/JPEG files; **Choose folder (includes subfolders)**
includes nested images. The user reviews the selected names/count and presses
**Start digitization**. Selection alone does not upload; **Clear selection**
resets the list. The GUI has no region-mode or OCR selectors. Manual-region mode
keeps the existing image picker and Plot area/Legend controls.
If the user already supplied a folder/files and scope in chat, preserve that
instruction and use the batch API without asking again. Never invent a folder.
Install Tesseract only when the manual-region route selects it, then verify it.

Create a local Python 3.12 virtual environment and install `requirements.txt`
using its Python. On Windows:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

On macOS/Linux use `python3.12 -m venv .venv` and
`.venv/bin/python -m pip install -r requirements.txt`. Installation was verified
on Windows; resolve platform-specific failures before claiming readiness.

After the user chooses AI regions, agent vision OCR is automatic. Launch:

```powershell
.\.venv\Scripts\python.exe run.py --region-mode agent --ocr-engine agent --port 8000
```

Use `--region-mode manual` when the user will draw the regions, and
`--ocr-engine tesseract` only for the manual-region route when Tesseract was chosen.
AI regions imply agent OCR; explicit Tesseract/fallback flags with AI regions
are rejected, including `--agent-workflow --ai-ocr`.
On macOS/Linux use `.venv/bin/python run.py` with the same flags. A plain
`run.py` starts manual regions and Tesseract every time, even after a previous
agent session. The user does not repeat those choices in the browser.

`--agent-workflow` remains an alias for `--region-mode agent` and conflicts with
explicit manual mode. `--ocr-engine auto-agent` tries Tesseract and requests
agent OCR only on missing/failed execution; it does not judge text accuracy.
`--ai-ocr` is a legacy alias for that fallback and conflicts with a different
explicit OCR engine. Explicit agent OCR is never replaced by Tesseract.
For a server already owned by this task, the agent can apply the resolved region/OCR
settings through `POST /agent-workflow/settings` and verify them through status.

Use a free port instead of stopping an unrelated process. Keep the server
running in the environment's supported background mechanism; on Windows start
helpers without an unwanted terminal window. Verify `/openapi.json` and
`/agent-workflow/status`, then give or open the exact local URL.

### Keep the operating agent active

Send `POST /agent-workflow/heartbeat` while operating the app. After the
region choice, launch and open the AI-mode picker. Keep polling
`GET /agent-workflow/status` for GUI-created batches and service
`GET /agent-workflow/requests` plus `GET /ocr/requests` as work arrives.
The GUI uploads selected images through `/agent-workflow/batches/upload` only
when the user presses **Start digitization**. It always uses `ocr_mode: "agent"`,
`region_mode: "agent"`, and the existing Automatic detector. There are no browser
controls for region mode or OCR. For an input scope already given
in chat, `/agent-workflow/batches` remains an alternative using the supplied
folder and optional relative `files` list.

After launch, stay active and send heartbeats while monitoring status for
GUI-created batches and servicing region/OCR requests. Do not finish with a
"ready" message while the user is selecting inputs. The GUI can queue images
while an agent is inactive, but displays queued/waiting status and never sends
agent heartbeats. A heartbeat must represent an agent actually available to
answer, not a heartbeat-only background process. The program cannot summon
this conversation or perform unattended AI vision. After every selected task
has a terminal state, report the results and return.

For manual regions with AI OCR, keep polling while the user uploads a chart,
draws regions and presses Run in the existing GUI. Do not block on a long
`/digitize` response before answering OCR.

View each actual chart before selecting regions and each actual OCR snippet
before answering. Copy request hashes and use the returned answer URLs.
Do not use ground truth, synthetic manifests, saved result JSON or expected
values to supply regions or OCR. The existing pipeline reads numeric ticks,
labels and units; preserve unknown or ambiguous values as unknown.
Read [src/AI_OCR.md](src/AI_OCR.md) for the exact text/word-box schema.

Keep servicing both queues and checking the batch until every selected image
has a terminal state. Requests appear sequentially; a temporary empty queue is
not completion. Report output paths and completed, failed and review counts.
Give each completed image its explicit review link in chat, using returned IDs:
`http://127.0.0.1:PORT/?batch=BATCH_ID&item=ITEM_ID`. The link loads only that
result in the existing GUI. The picker's **Review** action uses the same link;
**Download batch results (.zip)** provides the batch download. Never guess the newest job.
Portable image/JSON pairs still open through the existing Edit saved results
controls. Give the batch snapshot's `review_dir` so the user can select the flat
`review/` folder through **Edit saved results → Choose edit folder**, browse its
matching PNG/JSON pairs, edit and save. Retries preserve prior pairs with a job
suffix; the ZIP includes `review/`. Check each item's `review_error` before
claiming a pair is ready. Keep the explicit per-image links as well.
The application cannot call the current agent conversation itself. Image and
local HTTP/file tools are required; no API key or second assistant is needed.

### Manual selection and browser handoff

When the user chooses manual regions, open the running page and let the user
choose the image, press **Plot area**, and drag around the plot. Use **Legend**
for a visible legend. Use the application's canvas, not browser annotation mode.
After changing/preparing the image, reselect the regions. Selecting a rectangle
alone does not send it to the AI conversation; **Run** submits it to `/digitize`.
The workflow API queues separate explicit agent-region requests; it does not
read arbitrary unsent browser rectangles.

Choose the browser handoff according to the tools actually available:

| Environment | User-facing route |
| --- | --- |
| Codex desktop with an available built-in browser | Open the local URL visibly, for example with `open_in_codex` when exposed. |
| Codex CLI / an IDE without browser integration | Give the local URL for the user's Chrome/Edge browser. |
| Claude Code with browser integration | Use its configured connected browser. |
| Claude Code without browser integration | Give the local URL for the user to open manually. |

Check the current tools rather than assuming every CLI has a browser. The
[Codex browser documentation](https://learn.chatgpt.com/docs/browser?surface=app)
and [Claude Code Chrome guide](https://code.claude.com/docs/en/chrome) describe
their integrations. Let the user operate file pickers when the agent cannot.
WebFetch retrieves content; it cannot collect canvas drag selections or replace
viewing the actual image. A cloud fetcher may not reach the user's localhost.
Use supported port forwarding for remote servers: localhost belongs to the
machine running the browser.

### Regions, calibration and required review

All web rectangles use inclusive `x0,y0,x1,y1` working-image pixels and are
validated against the raster. The GUI's manual boxes remain `boxes.plot` and
`boxes.legend`; Colour uses `legend_box`, B&W uses `legend_area`, and Automatic
accepts either spelling while rejecting conflicts. Automatic requires a plot
area and resolves Colour/B&W from working-image pixels through
`automatic_mode_v46.select_detection_mode`. The resolved response `mode` is
`color` or `bw`; `requested_mode` and `mode_selection` preserve the choice and
evidence. Saved sessions and Step 5 retain the detector that produced them.
Do not confuse agent region selection with automatic detector selection.

A missing legend takes the existing one-series route. Do not invent extra
series. Human review/correction is required in the existing GUI: check the
plot/legend, labels, overlay, reconstructed plot, coordinate table, calibration
and units before using the data. Retain unknown/pixel calibration visibly.
Manual ranges refer to plot edges; use the optional scale box for interior
ticks. Preserve original images and image-linked JSON, IDs and evidence.

## Runtime map

| Component | Responsibility and connections |
| --- | --- |
| `src/ocr_bridge_v46.py` | Selected Tesseract/agent OCR and local snippet/answer queue; protocol in `src/AI_OCR.md` |
| `src/AI_WORKFLOW.md` | Startup choices, region and folder-batch API, active agent service loop, and human review |
| `src/unified_server.py` | FastAPI routes, AI OCR request/image/answer endpoints, image preparation, detector/correction child processes, job responses, saved-session import/export |
| `src/automatic_mode_v46.py` | Lightweight, model-free Colour/B&W selection from selected plot and legend pixels; never replaces manual overrides |
| `src/axis_calibration_v46.py` | Shared numeric tick OCR and conservative per-axis linear/log fitting; manual overrides and explicit pixel fallback |
| `src/axis_units_v46.py` | Independent axis-title OCR and conservative unit recognition; does not change calibrated values |
| `src/fast_detection_v46.py` | Optional 1,400-pixel working-image cap, inclusive region transforms, source/working provenance and original-resolution axis-reading context |
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

Blank X/Y limits request `axis_calibration_v46.resolve_calibration()` in the B&W,
colour and no-legend export paths. The server must omit blank CLI limits, not
replace them with 0 and 1. Each axis is independent: both manual limits skip numeric-tick OCR
and use linear mapping unless the log flag is set. Axis-title/unit OCR remains
independent and may still request an agent answer. A partial manual input is
preserved only when a compatible automatic fit supplies the missing bound.
Automatic fits need three distinct supporting labels and reject contradictory
or ambiguous scales. Failed axes retain editable marker pixels and export local
pixel distances with `kind: pixel`, `status: needs_calibration` and a warning.
Do not describe those distances as physical units or score them as estimated data.

`axis_calibration.json` and saved `edit_data.json` preserve calibration sources,
diagnostics and warnings. GUI result/table status displays the calibration;
imported automatic values must not silently become new manual overrides. Saved
Step-5 correction reuses calibration rather than rereading axis labels. The GUI
axis fields remain visible; the optional scale box still requires all four manual
limits and transforms them to the plot-edge convention. Preserve full numerical
precision in coordinate exports, especially small concentration values.

Axis-title unit recognition is independent of numeric calibration, including
when both bounds were entered manually. `axis_units_v46.read_axis_units()` uses
the selected OCR engine; it preserves readable unit text and explicit
`recognized`, `ambiguous`, `not_detected` or `unavailable` status. It does not
convert values between units or repair tick fits. `index.html` displays recognized
units in calibrated coordinate-table headings, with `axis units` when the unit
is unknown. Pixel/normalized headings stay explicitly uncalibrated.
`correction_session_v46.axis_export_units()` keeps physical units off
pixel/normalized values. CSV/Excel exports carry separate `x_unit`, `y_unit`,
`x_unit_status` and `y_unit_status` columns; saved JSON and Excel calibration
metadata retain the raw unit evidence even when calibration is unavailable.

Printed log-concentration values are returned in their printed units, not silently
exponentiated. Categorical/multiple-row axes, difficult exponents/decimals and
decreasing scales may need manual confirmation. Automatic detector mode selection
and automatic numeric axis calibration are independent features. The shared reader
uses the existing OCR bridge; it must stay import-safe without Tesseract or CUDA.

Resolution/rotation creates a new working image and invalidates incompatible
selections and results. Original image binding and prepared-image evidence are
separate. Match source fingerprints, dimensions, series identities and calibration
when importing; never accept an unrelated JSON just because the filename matches.

The optional GUI **Fast detection** checkbox is a separate next-Run preference,
off by default. `/digitize` resolves Automatic Colour/B&W using source pixels,
then `fast_detection_v46.prepare_job()` caps the longest side at 1,400 pixels.
The upload and selected source boxes remain unchanged; the server maps regions
and pairs every result, edit and Step-5 operation with the reduced working image.
Do not scale saved point coordinates a second time. Small images are not enlarged.
`fast_detection_v46.json` retains pixel hashes, dimensions and actual X/Y factors.
Axis OCR uses the hash-validated source image through a job-scoped environment
context, then maps only calibration pixel anchors. Numeric units and manual
overrides remain unchanged. Uncalibrated pixel distances remain source-image
pixel distances and must be labelled accordingly. Saved-only Step 5 reuses the
working image and calibration; the checkbox cannot resize an existing session.

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

The results page exposes a read-only **Extracted data — current result** table
without opening the editor. `coordinateRows()` and `drawCoordinateTable()` are
shared with the editor so calibration and formatting stay consistent. Preserve
calibration warnings, active/hypothesis labels, empty-series information and
stale-response guards. Suppressed candidates do not appear as extracted data.
Detection, Step 5, import and same-job edits update the current-result table.

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

AI OCR bridge verification: missing-Tesseract fallback, Tesseract preference in auto-agent mode,
word-box validation, image binding, timeout behavior and live HTTP answer handling
are covered by focused tests. Agent OCR is opt-in; Tesseract-only runs do not wait for an agent.
The old path-server test still expects v45 dispatch compatibility; that failure
was reproduced on the pre-OCR distribution and does not describe the v46-only contract.
