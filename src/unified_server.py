"""
Unified digitizer server — one FastAPI app, two modes.

  mode=color : v46 hybrid legend / automatic line-key source classification
  mode=bw    : v46 legend-grid B&W point detector
  no legend  : both modes -> v46 source-only single-series inference

Both modes end by writing the SAME edit_data.json (+ overlay + input.png) into the
job's out/ dir, so the front-end editor / live reconstruction / CSV are shared.

The v46 B&W point stage does not need torch, timm or trained model weights.
"""
from __future__ import annotations
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from starlette.concurrency import run_in_threadpool
import ocr_bridge_v46

HERE = Path(__file__).resolve().parent
JOBS_ROOT = Path(tempfile.gettempdir()) / "unified_digitizer_jobs"
SRC_DIR = Path(__file__).resolve().parent
JOBS_ROOT.mkdir(exist_ok=True)

# The colour pipeline's own post-detection outlier / fake-segment filters are
# disabled: the requested Step-5 correction replaces them. Flip this back to True to
# restore the built-in filters (it just stops setting NO_OUTLIER_FILTER).
COLOR_OUTLIER_FILTER = False
DETECTION_TIMEOUT_SECONDS = 900  # Multi-series native-resolution runs can exceed five minutes.

app = FastAPI(title="Unified Chart Digitizer")

# lazy singletons for the B&W path
_BW = {"run_detection": None, "adapter": None}
_OCR_JOBS = set()  # Only expose requests created by this server instance.


def _ocr_env(out_dir):
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    if ocr_bridge_v46.agent_enabled():
        _OCR_JOBS.add(out_dir.parent.name)
        env['CHARTOCODE_OCR_DIR'] = str((out_dir / 'ocr').resolve())
    return env


def _ocr_timeout(normal):
    return 3600 if ocr_bridge_v46.agent_enabled() else normal


def _ocr_directory(job_id):
    if job_id not in _OCR_JOBS:
        raise HTTPException(404, 'No AI OCR job with this ID on this server')
    return JOBS_ROOT / job_id / 'out' / 'ocr'


@app.get('/ocr/requests')
def ocr_requests():
    requests = []
    for job_id in tuple(_OCR_JOBS):
        for item in ocr_bridge_v46.pending_requests(_ocr_directory(job_id)):
            requests.append(dict(item, job_id=job_id,
                                 source_image_path=str(JOBS_ROOT / job_id / 'out' / 'input.png'),
                                 image_url=f'/ocr/{job_id}/{item["request_id"]}/image',
                                 answer_url=f'/ocr/{job_id}/{item["request_id"]}/answer'))
    return JSONResponse(dict(enabled=ocr_bridge_v46.agent_enabled(), requests=requests),
                        headers={'Cache-Control': 'no-store'})


@app.get('/ocr/{job_id}/{request_id}/image')
def ocr_image(job_id: str, request_id: str):
    directory = _ocr_directory(job_id)
    try:
        ocr_bridge_v46.read_request(directory, request_id)
    except (OSError, ValueError) as error:
        raise HTTPException(404, 'OCR snippet not found') from error
    return FileResponse(directory / 'snippets' / (request_id + '.png'), media_type='image/png')


@app.post('/ocr/{job_id}/{request_id}/answer')
async def ocr_answer(job_id: str, request_id: str, request: Request):
    directory = _ocr_directory(job_id)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 2 * 1024 * 1024:
            raise HTTPException(413, 'OCR answer exceeds 2 MiB')
    try:
        ocr_bridge_v46.submit_answer(directory, request_id, json.loads(body))
    except FileNotFoundError as error:
        raise HTTPException(404, 'OCR request not found') from error
    except (ValueError, TypeError) as error:
        raise HTTPException(400, str(error)) from error
    return {'accepted': True, 'request_id': request_id}


def _pick_color_pipeline() -> Path | None:
    """Use the supported v46 entry point; never fall back to older runners."""
    env = os.environ.get("PLOT_PIPELINE")
    pipeline = Path(env).expanduser().resolve() if env else HERE / "run_A4_auto_v46.py"
    if pipeline.name != "run_A4_auto_v46.py":
        raise ValueError("PLOT_PIPELINE must point to run_A4_auto_v46.py; legacy runners are unsupported")
    return pipeline if pipeline.is_file() else None


def _load_bw():
    """Import the B&W entry point + adapter on first use (heavy deps)."""
    if _BW["run_detection"] is None:
        sys.path.insert(0, str(HERE))
        import bw_pipeline               # GUI-free shared B&W runtime
        import bw_to_edit_data
        _BW["run_detection"] = bw_pipeline.run_detection
        _BW["adapter"] = bw_to_edit_data.bw_to_edit_data
    return _BW["run_detection"], _BW["adapter"]


def _parse4(s, field="region"):
    """An omitted rectangle is optional; an invalid submitted one is an error."""
    if not s.strip():
        return None
    try:
        values = tuple(float(v) for v in s.split(","))
        if len(values) != 4 or not all(math.isfinite(v) and v.is_integer() for v in values):
            raise ValueError("four finite integer coordinates required")
        box = tuple(int(v) for v in values)
        if min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("rectangle must have positive width and height")
        return box
    except (ValueError, OverflowError) as error:
        raise HTTPException(400, f"Invalid {field}: expected integer x0,y0,x1,y1 with positive width and height") from error


def _validate_uploaded_rois(in_path, regions):
    """Check submitted inclusive pixel coordinates against the actual raster."""
    try:
        raster = cv2.imread(str(in_path))
    except cv2.error as error:
        raise HTTPException(400, "Invalid image: could not decode the uploaded raster") from error
    if raster is None:
        raise HTTPException(400, "Invalid image: could not decode the uploaded raster")
    height, width = raster.shape[:2]
    for field, value in regions:
        box = _parse4(value, field)
        if box is not None and (box[2] >= width or box[3] >= height):
            raise HTTPException(400,
                f"Invalid {field} ROI {value!r} for image width={width}, height={height}: "
                f"inclusive coordinates require 0 <= x0 < x1 <= {width - 1} and "
                f"0 <= y0 < y1 <= {height - 1}. Reselect the region on the current image.")


def _optional_diagnostic(out_dir):
    path = out_dir / "legend_optional_v46.json"
    if not path.exists():
        return None
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError("diagnostic must be an object")
        return result
    except (OSError, ValueError):
        return {"status": "failed", "reason": "Could not read the v46 series inference diagnostic."}


def _triangle_errorbar_diagnostic(out_dir):
    """Small initial-detection summary, not the full candidate arrays."""
    path = out_dir / 'color_marker_hybrid_v46.json'
    if not path.exists():
        return None
    try:
        report = json.loads(path.read_text(encoding='utf-8'))
        guard = report.get('triangle_errorbar_guard')
        if not isinstance(guard, dict):
            return None
        return dict(guard, guide_policy=report.get('guide_policy'),
                    detector_status=report.get('status'), stage='initial_detection')
    except (OSError, ValueError, AttributeError):
        return dict(status='unavailable', reason='Triangle/error-bar diagnostic could not be read')


def _truthy(s):
    return s.strip().lower() in ("1", "true", "on", "yes")


@app.get("/", response_class=HTMLResponse)
def index():
    # Serve index.html from the SAME folder as this server (src/) first, so you can
    # keep everything in one place. Falls back to ../webapp/index.html if present.
    idx = HERE / "index.html"
    if not idx.exists():
        idx = HERE.parent / "webapp" / "index.html"
    return idx.read_text(encoding="utf-8") if idx.exists() else "<h1>index.html not found</h1>"


@app.api_route('/favicon.ico', methods=['GET', 'HEAD'], status_code=204, include_in_schema=False)
def favicon():
    # No custom tab icon is provided; acknowledge the browser's automatic request.
    return Response(status_code=204, headers={'Cache-Control': 'public, max-age=86400'})


@app.get('/image_sidecar_v46.js')
def image_sidecar_script():
    return FileResponse(str(HERE/'image_sidecar_v46.js'),media_type='text/javascript')


@app.post("/digitize")
async def digitize(
    image: UploadFile = File(...),
    mode: str = Form("color"),                 # "color" | "bw"
    # shared
    plot_area: str = Form(""),
    x_min: str = Form(""), x_max: str = Form(""),
    y_min: str = Form(""), y_max: str = Form(""),
    x_log: str = Form(""), y_log: str = Form(""),
    # colour-only
    legend_box: str = Form(""),
    color_series_mode: str = Form("auto"),     # infer markers or line-only from plot evidence
    # bw-only
    legend_area: str = Form(""),
    known_classes: str = Form(""),             # deprecated; accepted but ignored for BW v46
    has_errorbars: str = Form(""),             # deprecated BW field; accepted but ignored
    conf: str = Form(""),
    scale: str = Form(""),                     # deprecated BW detector scale; accepted but ignored
    correct: str = Form(""),                   # "", "1" = run mode-specific Step-5 correction
    correct_iters: str = Form(""),             # "" = default cap; else max iteration count
    prev_job: str = Form(""),                  # previous job id -> continue Step-5 from its state
):
    job_id = uuid.uuid4().hex[:12]
    job_dir = JOBS_ROOT / job_id
    out_dir = job_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    in_path = job_dir / "input.png"
    with in_path.open("wb") as f:
        shutil.copyfileobj(image.file, f)
    legend_field, legend_value = ("legend_area", legend_area) if mode == "bw" else ("legend_box", legend_box)
    _validate_uploaded_rois(in_path, (("plot_area", plot_area), (legend_field, legend_value)))
    # input copy the front-end can display
    shutil.copyfile(in_path, out_dir / "input.png")

    try:
        if mode == "bw":
            return await run_in_threadpool(_run_bw, job_id, in_path, out_dir, plot_area, legend_area,
                           known_classes, x_min, x_max, y_min, y_max, x_log, y_log,
                           has_errorbars, conf, correct, scale, correct_iters, prev_job)
        return await run_in_threadpool(_run_color, job_id, in_path, out_dir, plot_area, legend_box,
                          x_min, x_max, y_min, y_max, x_log, y_log,
                          correct, correct_iters, prev_job, color_series_mode)
    except subprocess.TimeoutExpired as error:
        # TimeoutExpired retains bytes even when subprocess text mode is used.
        output = ''.join(value.decode('utf-8', errors='replace') if isinstance(value, bytes)
                         else value or '' for value in (error.stdout, error.stderr))
        message = (f'Processing timed out after {error.timeout:g} seconds. '
                   'The job was stopped before completion. Try a lower image resolution '
                   'under Image preparation, then redraw the plot and legend areas.')
        response = json.loads(_response(job_id, out_dir, mode, False, output + '\n' + message).body)
        response.update(summary=message, timed_out=True)
        return JSONResponse(response)


@app.post('/prepare-image')
def prepare_image(
    image: UploadFile = File(...),
    resolution_percent: str = Form('100'),
    rotation_degrees: str = Form('0'),
):
    """Shared Colour/B&W UI preparation; process the original in memory.

    Positive rotation is clockwise. The client uses the returned raster as its
    new coordinate system and must clear/reselect plot/legend boxes afterward.
    Existing detection jobs and their images are never touched by this route.
    """
    from bw_image_preparation import MAX_UPLOAD_BYTES, ImagePreparationError, prepare_image_bytes
    payload = image.file.read(MAX_UPLOAD_BYTES + 1)
    try:
        png, width, height = prepare_image_bytes(payload, resolution_percent, rotation_degrees)
    except ImagePreparationError as error:
        raise HTTPException(error.status_code, str(error)) from error
    return Response(png, media_type='image/png', headers={
        'X-Image-Width': str(width), 'X-Image-Height': str(height), 'Cache-Control': 'no-store'})


def _run_color(job_id, in_path, out_dir, plot_area, legend_box,
               x_min, x_max, y_min, y_max, x_log, y_log,
               correct="", correct_iters="", prev_job="", color_series_mode="auto"):
    if color_series_mode not in ('auto', 'markers', 'line-only'):
        raise HTTPException(400, 'Unknown colour series mode')
    legend = _parse4(legend_box, "legend_box")
    plot = _parse4(plot_area, "plot_area")
    if legend is None and plot is None:
        raise HTTPException(400, "No legend selected: draw a plot area to infer one series")
    # A missing user selection explicitly selects v46 single-series inference.
    try:
        pipeline = _pick_color_pipeline()
    except ValueError as error:
        raise HTTPException(400, str(error)) from error
    if pipeline is None:
        raise HTTPException(500, "run_A4_auto_v46.py is missing; restore the complete v46 runtime")
    cmd = [sys.executable, str(pipeline), str(in_path), str(out_dir),
           '--mode', 'color', '--color-series-mode', color_series_mode]
    if legend is None: cmd += ['--no-legend']
    if legend_box.strip(): cmd += ["--legend-box", legend_box.strip()]
    if plot_area.strip():  cmd += ["--plot-area", plot_area.strip()]
    if x_min.strip():      cmd += ["--x-min", x_min.strip()]
    if x_max.strip():      cmd += ["--x-max", x_max.strip()]
    if y_min.strip():      cmd += ["--y-min", y_min.strip()]
    if y_max.strip():      cmd += ["--y-max", y_max.strip()]
    if _truthy(x_log):     cmd += ["--x-log"]
    if _truthy(y_log):     cmd += ["--y-log"]
    _env = _ocr_env(out_dir)
    if not COLOR_OUTLIER_FILTER:
        _env["NO_OUTLIER_FILTER"] = "1"
    print(f"[unified] colour job: correct={correct!r} iters={correct_iters!r} "
          f"prev_job={prev_job!r}  outlier_filter="
          f"{'ON' if COLOR_OUTLIER_FILTER else 'OFF (replaced by Step-5)'}", flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                          timeout=_ocr_timeout(DETECTION_TIMEOUT_SECONDS), env=_env)
    _log = (proc.stdout or "") + (proc.stderr or "")
    # Surface the pipeline's own stdout (incl. DEBUG_LABELS [labels] lines) to the
    # server console. capture_output=True otherwise swallows it into the buffer, so
    # nothing prints in the terminal even when DEBUG_LABELS=1 is set.
    print(_log, flush=True)
    ok = proc.returncode == 0 and (out_dir / "edit_data.json").exists()
    diagnostic = _optional_diagnostic(out_dir)
    if diagnostic is not None and diagnostic.get('status') != 'completed':
        ok = False

    # v46 colour uses path-shape correction, P0 protection and saved hypothesis
    # identity. Every supported colour run uses the same v46 correction contract.
    if not _truthy(correct):
        print("[unified] colour Step-5 not requested (correct not set)", flush=True)
    if ok and _truthy(correct) and (diagnostic or {}).get('correction_available') is False:
        _log += '\n[color Step-5 unavailable; detection output preserved] ' + diagnostic.get(
            'correction_unavailable_reason', 'No validated correction reference is available.') + '\n'
        ok = False
    if ok and _truthy(correct):
        ccmd = [sys.executable, str(SRC_DIR / "color_correct_cli.py"),
                str(in_path), str(out_dir)]
        ccmd += ['--require-v46-path']
        if plot_area.strip():    ccmd += ["--plot-area", plot_area.strip()]
        if legend_box.strip():   ccmd += ["--legend-area", legend_box.strip()]
        if correct_iters.strip():ccmd += ["--correct-iters", correct_iters.strip()]
        if prev_job.strip():
            _prev = JOBS_ROOT / prev_job.strip() / "out" / "color_correction_state.json"
            if _prev.exists():
                ccmd += ["--prev-state", str(_prev)]
        print("[unified] colour Step-5:", " ".join(ccmd), flush=True)
        cproc = subprocess.run(ccmd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                               timeout=_ocr_timeout(900), env=_env)
        _clog = (cproc.stdout or "") + (cproc.stderr or "")
        print(_clog, flush=True)
        _log += "\n---- colour Step-5 ----\n" + _clog
        if cproc.returncode != 0:
            # A valid detection file is not evidence that correction succeeded.
            # Keep detection artifacts for inspection, but surface the failure.
            _log += f'\n[color Step-5 FAILED, exit {cproc.returncode}; detection output preserved]\n'
            ok = False

    return _response(job_id, out_dir, "color", ok, _log)


# Legacy editor-compatible class names, not a web input filter. v46 takes all
# automatically extracted legend entries, including unknown_marker, unchanged.
BW_ALL_CLASSES = [
    "filled_circle", "open_circle", "filled_square", "open_square",
    "filled_triangle", "open_triangle", "filled_inv_triangle", "open_inv_triangle",
    "filled_rhombus", "open_rhombus", "x_marker", "plus_marker",
]


def _run_bw(job_id, in_path, out_dir, plot_area, legend_area, known_classes,
            x_min, x_max, y_min, y_max, x_log, y_log, has_errorbars, conf,
            correct="", scale="", correct_iters="", prev_job=""):
    """Run v46 B&W on the submitted prepared raster without a second resize.

    Deprecated shape, errorbar and detector-scale form fields remain accepted
    for old clients, but do not change the current web detection workflow.
    """
    if _parse4(plot_area, "plot_area") is None:
        raise HTTPException(400, "bw mode requires plot_area = x0,y0,x1,y1")
    legend = _parse4(legend_area, "legend_area")
    cli = HERE / "run_A4_auto_v46.py"
    if not cli.exists():
        raise HTTPException(500, "run_A4_auto_v46.py not found next to unified_server.py")
    cmd = [sys.executable, str(cli), str(in_path), str(out_dir),
           "--mode", "bw", "--skip-segments",
           "--plot-area", plot_area.strip()]
    for flag, value, default in (("--x-min", x_min, "0"), ("--x-max", x_max, "1"),
                                 ("--y-min", y_min, "0"), ("--y-max", y_max, "1")):
        # New single-series inference keeps pixel units when no scale was supplied.
        if value.strip() or legend is not None:
            cmd += [flag, value.strip() or default]
    if legend is None:       cmd += ["--no-legend"]
    if legend_area.strip():   cmd += ["--legend-area", legend_area.strip()]
    if _truthy(x_log):        cmd += ["--x-log"]
    if _truthy(y_log):        cmd += ["--y-log"]
    # Older web clients may still submit a manual shape list. Keep the form
    # field accepted for compatibility, but never let it filter automatically
    # extracted legend series (especially unknown_marker). The CLI's explicit
    # --classes option remains available outside this web GUI.
    if known_classes.strip():
        print("[unified] BW: ignored deprecated known_classes; using automatic legend classification", flush=True)
    if conf.strip():          cmd += ["--conf", conf.strip()]
    # Marker detection tolerates connecting/errorbar ink as occlusion, so the
    # point stage does not erase those pixels. Requested SSIM Step-5 separately
    # runs the v45 segment extractor on its own comparison reference image.
    if has_errorbars.strip():
        print('[unified] BW: ignored deprecated has_errorbars; segment/errorbar detection is disabled', flush=True)
    if scale.strip():
        print('[unified] BW: ignored deprecated detector scale; using the prepared image at 1x', flush=True)
    if _truthy(correct):      cmd += ["--correct"]
    if correct_iters.strip(): cmd += ["--correct-iters", correct_iters.strip()]
    if _truthy(correct) and prev_job.strip():
        # continue Step-5 from the previous job's saved correction state
        previous_out = JOBS_ROOT / prev_job.strip() / "out"
        previous_diagnostic = _optional_diagnostic(previous_out)
        state_name = ('color_correction_state.json' if previous_diagnostic and
                      previous_diagnostic.get('chart_kind') == 'line-only' else 'correction_state.json')
        _prev = previous_out / state_name
        if not _prev.exists():
            try:
                previous_status = json.loads((previous_out / 'correction_status.json').read_text(encoding='utf-8'))
            except FileNotFoundError:
                previous_status = {'status': 'not_requested', 'requested': False}
            except (OSError, ValueError):
                previous_status = {}
            first_type3_correction = (state_name == 'color_correction_state.json' and
                                     previous_diagnostic.get('status') == 'completed' and
                                     isinstance(previous_status, dict) and
                                     previous_status.get('status') == 'not_requested' and
                                     previous_status.get('requested') is False)
            if not first_type3_correction:
                raise HTTPException(409, 'Previous B&W correction state is unavailable; run detection again.')
        else:
            cmd += ["--prev-state", str(_prev)]

    print("[unified] BW subprocess:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=_ocr_timeout(DETECTION_TIMEOUT_SECONDS), env=_ocr_env(out_dir))
    # Echo the child's output to the server console so failures are visible.
    if proc.stdout:
        print("---- bw_detect_cli stdout ----\n" + proc.stdout)
    if proc.stderr:
        print("---- bw_detect_cli stderr ----\n" + proc.stderr)
    print(f"---- bw_detect_cli exit code: {proc.returncode} ----")
    ok = proc.returncode == 0 and (out_dir / "edit_data.json").exists()
    diagnostic = _optional_diagnostic(out_dir)
    if diagnostic is not None and diagnostic.get('status') != 'completed':
        ok = False
    log = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if _truthy(correct):
        try:
            status = json.loads((out_dir / 'correction_status.json').read_text(encoding='utf-8'))
            expected_engines = ('bw_step5_v46', 'bw_series_correction_v46')
            if diagnostic and diagnostic.get('chart_kind') == 'line-only':
                expected_engines += ('v46_type3_path_v1',)
            completed = status.get('status') == 'completed' and status.get('engine') in expected_engines
        except (OSError, ValueError):
            completed = False
        if not completed or not ok:
            ok = False
            log += ('\n[B&W Step-5 FAILED or incomplete; any detection artifacts are '
                    'preserved for inspection and are not a successful correction result.]\n')
    return _response(job_id, out_dir, "bw", ok, log)


def _response(job_id, out_dir, mode, ok, log):
    (out_dir / 'pipeline.log').write_text(log, encoding='utf-8')
    def url(name):
        p = out_dir / name
        return f"/result/{job_id}/{name}" if p.exists() else None
    diagnostic = _optional_diagnostic(out_dir)
    editable = False
    editing_reason = 'No editable series were saved. Adjust the plot/legend selection and run detection again.'
    if (out_dir/'edit_data.json').exists() and (out_dir/'input.png').exists():
        from correction_session_v46 import read, validate_edit
        try:
            raster = cv2.imread(str(out_dir/'input.png'))
            if raster is None:
                raise ValueError('The paired image could not be decoded')
            validate_edit(read(out_dir/'edit_data.json'), raster)
            editable, editing_reason = True, None
        except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
            editing_reason = 'Saved data cannot be edited: '+str(error)
    if diagnostic is not None:
        ok = bool(ok) and diagnostic.get('status') == 'completed'
    meta_path = out_dir / 'session_meta.json'
    if not meta_path.exists() and (ok or editable):
        meta_path.write_text(json.dumps({'mode':mode}),encoding='utf-8')
    meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
    if ok or editable:
        from correction_history_v46 import now
        meta.setdefault('version_id',job_id)
        meta.setdefault('version_kind','imported' if meta.get('imported') else 'detection')
        if not meta.get('created_at'):
            meta['created_at'] = now()
        if not ok and editable:
            meta.setdefault('manual_only_reason', 'Automatic detection did not complete. Saved series can be edited manually; Step 5 requires a completed detection with usable correction evidence.')
        meta_path.write_text(json.dumps(meta),encoding='utf-8')
    triangle_guard = _triangle_errorbar_diagnostic(out_dir) if mode == 'color' else None
    group_backend = (out_dir/'color_group_state_v46.json').exists()
    # Advertise only files that exist. The BW renderer uses different artifacts
    # from the legacy colour table diagnostic; the UI must not guess filenames.
    if mode == 'bw':
        legend_diagnostic = url('v46_legend_diagnostic.png') or url('v46_legend_templates.png')
        legend_overlay = url('v46_legend_overlay.png')
        legend_kind = 'bw-observed-templates'
    else:
        legend_diagnostic = (url('color_marker_templates_v46.png') if group_backend else None) or url('legend_diagnostic.png')
        legend_overlay = url('legend_overlay.png')
        legend_kind = 'color-group-templates' if group_backend else 'color-table'
    summary = "Detection complete." if ok else "Detection or correction failed; inspect the diagnostic and log."
    if diagnostic is not None:
        ok = ok and diagnostic.get('status') == 'completed'
        kind = diagnostic.get('chart_kind', 'undetermined')
        reason = diagnostic.get('reason') or 'No diagnostic reason supplied.'
        summary = f"v46 series inference: {diagnostic.get('status', 'unknown')} ({kind}). {reason}"
        if 'points' in diagnostic:
            active = len(diagnostic.get('points') or [])
            suppressed = len(diagnostic.get('suppressed_points') or [])
            summary += f' {active} active points; {suppressed} suppressed hypotheses.'
            if active == 0 and diagnostic.get('status') == 'completed':
                summary += ' No active measurement points; curve evidence is available for correction.'
        if diagnostic.get('calibration_warning'):
            summary += ' ' + diagnostic['calibration_warning']
        if diagnostic.get('correction_unavailable_reason'):
            summary += ' ' + diagnostic['correction_unavailable_reason']
        if not ok and diagnostic.get('status') == 'completed':
            summary += ' Detection or correction failed; inspect the log.'
    export_error = None
    if ok or editable:
        try:
            from correction_session_v46 import save_outputs
            save_outputs(out_dir,mode)
        except Exception as error:
            export_error = 'JSON/Excel export failed: '+str(error)
            summary += ' '+export_error
    stop_path = out_dir / 'step5_stop_report.json'
    try:
        stop_report = json.loads(stop_path.read_text(encoding='utf-8')) if ok and stop_path.exists() else None
        if not isinstance(stop_report,dict): stop_report = None
    except (OSError,ValueError):
        stop_report = None
    from color_step5_defaults_v46 import settings_summary
    colour_objective=settings_summary(out_dir) if mode=='color' else None
    return JSONResponse({
        "job_id": job_id, "mode": mode, "ok": ok,
        "editable_available": editable,
        "editing_unavailable_reason": editing_reason,
        "result_version": {k:meta.get(k) for k in ('version_id','version_kind','created_at','iterations','stop_policy')},
        "summary": summary,
        "correction_backend": 'colour_group_bw_elements' if group_backend else None,
        "step5_stop": stop_report,
        "color_step5_objective": colour_objective,
        "step5_stop_url": url('step5_stop_report.json') if stop_report else None,
        "legend_optional": diagnostic,
        "legend_optional_url": url("legend_optional_v46.json"),
        "triangle_errorbar": triangle_guard,
        "triangle_errorbar_overlay_url": url('triangle_errorbar_review_v46.png') if triangle_guard else None,
        "triangle_errorbar_weak_url": url('triangle_errorbar_weak_suppressed_v46.json') if triangle_guard else None,
        "triangle_errorbar_diagnostic_url": url('color_marker_hybrid_v46.json') if triangle_guard else None,
        "symbol_scale_diagnostic_url": url('color_symbol_scale_v46.json') if mode == 'color' else None,
        "legend_diagnostic_url": legend_diagnostic,
        "legend_overlay_url": legend_overlay,
        "legend_diagnostic_kind": legend_kind if legend_diagnostic or legend_overlay else None,
        "step5_available": bool(ok) and (diagnostic or {}).get('correction_available', True) and
                           ('engine' not in meta or meta['engine'] is not None),
        "correction_only": bool(meta.get('detection_skipped')),
        "correction_unavailable_reason": meta.get('manual_only_reason') or
            (diagnostic or {}).get('correction_unavailable_reason') or
            (None if ok else 'Automatic detection did not complete; review the analysis and adjust the plot/legend selection before Step 5.'),
        "paired_image_url": f'/correction-image/{job_id}' if editable else None,
        "edit_data_url": url("edit_data.json"),
        "input_url": url("input.png"),
        "overlay_url": url("data_points_overlay.png"),
        "xlsx_url": url("current_data.xlsx") if not export_error else None,
        "correction_session_url": url("correction_session.json") if not export_error else None,
        "export_error": export_error,
        "log": log[-4000:],
        "log_url": url('pipeline.log'),
    })


def _job_output(job_id):
    import re
    if not re.fullmatch(r'[a-f0-9]{12}',str(job_id)):
        raise HTTPException(400,'Invalid job ID')
    folder = JOBS_ROOT / job_id / 'out'
    if not folder.is_dir() or folder.is_symlink():
        raise HTTPException(404,'Saved job is unavailable; load the paired image and correction JSON')
    return folder


async def _session_request(request):
    from correction_session_v46 import MAX_JSON, loads
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks)+len(chunk) > MAX_JSON:
            raise HTTPException(413,'Correction request exceeds 128 MB')
        chunks.extend(chunk)
    try:
        data = loads(chunks)
        if not isinstance(data,dict):
            raise ValueError('Expected a JSON object')
        return data
    except (ValueError,TypeError,RecursionError) as error:
        raise HTTPException(400,str(error)) from error


@app.post('/correction-session/import')
def import_correction_session(image: UploadFile = File(...), data_file: UploadFile = File(...),
                              mode: str = Form('color'), version_id: str = Form('')):
    from correction_session_v46 import import_session, MAX_IMAGE, MAX_JSON
    job = uuid.uuid4().hex[:12]
    out = JOBS_ROOT / job / 'out'
    try:
        info = import_session(image.file.read(MAX_IMAGE+1),data_file.file.read(MAX_JSON+1),out,mode,version_id or None)
    except (ValueError,TypeError,KeyError,OSError,AttributeError,RecursionError) as error:
        raise HTTPException(400,'Cannot load correction pair: '+str(error)) from error
    response = json.loads(_response(job,out,info['mode'],True,'Imported saved coordinates/evidence. Detection was not run.').body)
    response.update(operation='import',summary='Loaded image + saved data. Detection skipped. '+
        (info['reason'] or 'Manual editing and saved-evidence Step 5 are available.')+
        ('' if info['paired'] else ' Legacy JSON: only image dimensions could be checked; verify the overlay.'),
        imported_edit_data_url=response['edit_data_url'],loaded_version=info.get('version'))
    return JSONResponse(response)


@app.post('/correction-session/export')
async def export_correction_session(request: Request):
    from correction_session_v46 import export_session, read
    data = await _session_request(request)
    folder = _job_output(data.get('job_id'))
    meta = read(folder/'session_meta.json') if (folder/'session_meta.json').exists() else {}
    try:
        saved = export_session(folder,meta.get('mode',data.get('mode','color')),data.get('edit_data'))
    except (ValueError,TypeError,KeyError,OSError,AttributeError) as error:
        raise HTTPException(400,str(error)) from error
    return JSONResponse(saved,headers={'Content-Disposition':f'attachment; filename="correction_{data["job_id"]}.json"'})


@app.get('/correction-image/{job_id}')
def download_correction_image(job_id: str):
    folder = _job_output(job_id)
    image = cv2.imread(str(folder/'input.png'))
    if image is None:
        raise HTTPException(404,'Paired image is unavailable')
    ok, png = cv2.imencode('.png',image)
    if not ok:
        raise HTTPException(500,'Cannot encode paired PNG')
    return Response(png.tobytes(),media_type='image/png',headers={
        'Content-Disposition':f'attachment; filename="correction_{job_id}.png"'})


def _read_export_edits(edit_data):
    """Accept legacy small text fields and bounded JSON file uploads.

    Multipart text fields have a separate 1 MiB parser limit. The GUI sends
    edit_data as a file so large snapshots use the 128 MiB application limit.
    """
    from correction_session_v46 import loads, MAX_JSON
    payload = edit_data.encode('utf-8') if isinstance(edit_data, str) else edit_data.file.read(MAX_JSON+1)
    return loads(payload)


@app.post('/correction-session/export-linked')
def export_linked_correction(original: UploadFile = File(...), job_id: str = Form(...),
                             edit_data: UploadFile | str = File('null')):
    from correction_session_v46 import export_linked_session, read, MAX_IMAGE
    folder = _job_output(job_id)
    try:
        proposed = _read_export_edits(edit_data)
        meta = read(folder/'session_meta.json')
        package = export_linked_session(folder,meta['mode'],original.file.read(MAX_IMAGE+1),
                                        original.filename or 'plot.png',proposed)
    except (ValueError,TypeError,KeyError,OSError,AttributeError) as error:
        raise HTTPException(400,str(error)) from error
    return JSONResponse(package)


@app.post('/correction-session/excel')
async def export_current_excel(request: Request):
    from correction_session_v46 import edited_data, excel_bytes, read
    data = await _session_request(request)
    folder = _job_output(data.get('job_id'))
    try:
        image = cv2.imread(str(folder/'input.png'))
        if image is None:
            raise ValueError('Paired job image is unavailable')
        ed = edited_data(read(folder/'edit_data.json'),data.get('edit_data'),image)
        workbook = excel_bytes(ed)
    except (ValueError,TypeError,KeyError,OSError,AttributeError,OverflowError) as error:
        raise HTTPException(400,str(error)) from error
    return Response(workbook,media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={'Content-Disposition':f'attachment; filename="data_{data["job_id"]}.xlsx"'})


@app.post('/correction-session/export-history')
def export_correction_history(original: UploadFile = File(...), job_id: str = Form(...),
                              edit_data: UploadFile | str = File('null')):
    from correction_session_v46 import MAX_IMAGE
    from correction_history_v46 import export_history
    try:
        package = export_history(_job_output(job_id),original.file.read(MAX_IMAGE+1),
                                 original.filename or 'plot.png',_job_output,_read_export_edits(edit_data))
    except (ValueError,TypeError,KeyError,OSError,AttributeError,RecursionError) as error:
        raise HTTPException(400,str(error)) from error
    return JSONResponse(package)


@app.post('/correct-saved')
async def correct_saved_session(request: Request):
    """Fork a saved job and run only correction. The original job is immutable."""
    from correction_session_v46 import export_session, import_session, read, save
    data = await _session_request(request)
    source = _job_output(data.get('job_id'))
    from step5_stop_v46 import resolve_policy
    try:
        config = resolve_policy(data.get('stop_policy','manual'), data.get('iterations',5))
    except ValueError as error:
        raise HTTPException(400,str(error)) from error
    iterations = config['max_iterations']
    meta = read(source/'session_meta.json') if (source/'session_meta.json').exists() else {}
    job = uuid.uuid4().hex[:12]
    out = JOBS_ROOT / job / 'out'
    try:
        package = export_session(source,meta.get('mode',data.get('mode','color')),data.get('edit_data'))
        info = import_session((source/'input.png').read_bytes(),json.dumps(package,allow_nan=False).encode(),out)
        if not info['engine']:
            raise ValueError(info['reason'])
        from correction_history_v46 import now
        child_meta = read(out/'session_meta.json')
        child_meta.update(parent_job=data['job_id'],version_id=job,version_kind='step5',
                          created_at=now(),iterations=iterations,stop_policy=config['policy'])
        save(out/'session_meta.json',child_meta)
        save(out/'step5_start_ed.json',package['edit_data'])
    except (ValueError,TypeError,KeyError,OSError,AttributeError) as error:
        raise HTTPException(400,str(error)) from error
    cmd = [sys.executable,str(SRC_DIR/'correction_only_cli_v46.py'),str(out),'--iterations',str(iterations),
           '--stop-policy',config['policy']]
    # Await the subprocess without blocking the async HTTP event loop.
    from starlette.concurrency import run_in_threadpool
    try:
        proc = await run_in_threadpool(subprocess.run,cmd,capture_output=True,text=True,
                                      encoding='utf-8',errors='replace',timeout=_ocr_timeout(900),env=_ocr_env(out))
        log = (proc.stdout or '')+(proc.stderr or '')
        status = read(out/'correction_only_status.json') if (out/'correction_only_status.json').exists() else {}
        ok = proc.returncode == 0 and status.get('completed') is True
    except subprocess.TimeoutExpired:
        log = 'Correction timed out. Original imported data are unchanged.'
        ok = False
    if not ok:
        # Failed work remains a separate job; never replace the user's input.
        save(out/'edit_data.json',package['edit_data'])
    response = json.loads(_response(job,out,info['mode'],ok,log).body)
    response.update(operation='correction',summary='Step 5 complete — detection skipped.' if ok else
                    'Correction failed; previous job and manual edits are preserved.',parent_job=data['job_id'])
    if ok and response.get('step5_stop'):
        response['summary'] += ' '+response['step5_stop']['summary']
    return JSONResponse(response)


@app.get("/result/{job_id}/{filename}")
def result(job_id: str, filename: str):
    path = JOBS_ROOT / job_id / "out" / filename
    if not path.exists():
        raise HTTPException(404, "not found")
    return FileResponse(str(path))


if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description='Start the v46 chart editor')
    parser.add_argument('--ai-ocr', action='store_true', help='Use the operating AI agent when Tesseract is unavailable')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.ai_ocr:
        os.environ['CHARTOCODE_OCR_MODE'] = 'auto-agent'
    print("colour pipeline:", _pick_color_pipeline())
    uvicorn.run(app, host=args.host, port=args.port)
