# Local AI OCR handoff (v46)

OCR means reading text from an image. All v46 runtime OCR calls use
`ocr_bridge_v46.py`. The selected engine determines who reads the text:
`tesseract` uses the external executable, `agent` always requests the operating
agent's image reading, and `auto-agent` tries Tesseract before using the agent
when the executable is unavailable or its call fails. Agent requests save the
exact image snippet as a PNG and wait for the operating Codex/Claude agent.
The agent reads the PNG with its image tool, submits JSON, and the **same
waiting operation continues** with that text or those word boxes.

This is a local handoff, not a bundled AI model. The agent must actively service
requests with image-viewing and local file/HTTP tools. Python cannot invoke the
current conversation's vision capability by itself. A separately configured
host scheduler may resume the operating agent for queued AI-mode batches, as
described under **Automatic chat wake-up** in [AI_WORKFLOW.md](AI_WORKFLOW.md).
The same OCR API works across compatible agents; waking an idle conversation
requires host-specific support and is not provided by the OCR bridge.
No API key, recursive
assistant CLI, model download or paid vision API is used. WebFetch alone cannot
view local files reliably or perform this workflow.

## Start and operate

First follow [AI_WORKFLOW.md](AI_WORKFLOW.md). Ask in the conversation whether
the user will draw regions or wants the agent to identify them. AI regions
automatically use agent vision OCR; do not ask an OCR question on that route.
Only for manual regions, ask whether OCR should use/install Tesseract or use
agent vision, unless already answered. Apply the choice through CLI/API; the
GUI has no OCR or region-mode selector. In agent-region
mode the user chooses multiple PNG/JPEG files or a folder (including subfolders)
in the AI-mode picker, reviews the file list and starts the batch. Do not ask a
path question. Preserve a folder/file scope already supplied in chat as
an optional direct API route. Do not silently change either workflow choice.

Packaged app with explicit AI OCR: `python run.py --ocr-engine agent --port 8000`.
Research source: `python src/unified_server.py --ocr-engine agent --host 127.0.0.1 --port 8000`.
Add `--region-mode agent` when the agent will identify regions; use
`--region-mode manual` when the user will draw them. `--agent-workflow` remains
a compatibility alias for agent regions. Plain `run.py` starts manual regions
with Tesseract, regardless of prior stored choices. Use
`--ocr-engine tesseract` for manual regions with Tesseract. For manual regions,
`--ocr-engine auto-agent` is an optional Tesseract-first fallback; the legacy
`--ai-ocr` flag retains that fallback. AI regions imply `--ocr-engine agent`;
explicit Tesseract/fallback flags with AI regions are rejected.

Keep the server and agent running. In manual-region mode the user uploads a
chart, draws Plot area and Legend, and presses Run. Agent-region and folder
batch operation use the same OCR queue; their image/region protocol is in
[AI_WORKFLOW.md](AI_WORKFLOW.md). After an agent-region launch, remain active
and monitor status/requests while the user selects GUI inputs and starts the
batch, unless a supported automatic wake-up has already been registered and
verified. Once a job starts, remain active through its terminal state rather
than waiting for the next scheduled wake-up to answer OCR.
Heartbeats must reflect an agent actually available to answer; do not
leave a heartbeat-only process claiming unattended AI availability. Report and
return when all selected tasks have terminal states. Start the server/long-running HTTP
request in the environment's supported background process so the agent remains
free to answer OCR. Do not block waiting for `/digitize` before checking OCR.
Use a free port instead of stopping the user's existing server.

1. Send `POST /agent-workflow/heartbeat` with `{"agent_id":"YOUR_SESSION_ID"}`
   while working, including during long image inspection. Heartbeats remain
   current for 90 seconds; sending one about every 30 seconds prevents a stale
   agent status. Poll `GET http://127.0.0.1:8000/ocr/requests` while the operation runs. Use
   short tool calls and reasonable polling intervals (for example 1–2 seconds).
   The response contains `enabled` and a `requests` list for this server only.
   Each request identifies its `job_id`, original `source_image_path`, snippet
   `image_path` / `image_url`, dimensions, `operation`, `config`, `request_id`,
   `image_sha256`, deadline and `answer_url`. Match the source image to the
   user's current operation; never answer an unrelated job.
2. Open the **actual snippet** using an image-viewing tool, or fetch `image_url`
   locally through a tool that can display images. Coordinates below are pixels
   of that snippet, including any existing padding/upscaling, not the full plot.
3. Read only visible text, retaining signs, decimal points, exponents, units and
   Greek letters. Use the requested Tesseract `config` as context (for example,
   a numeric tick crop), not as permission to invent missing characters. Inspect
   the actual chart as well when identifying its axes, titles and units. Do not
   copy ground truth, example manifests, previous result JSON, filenames or
   reference data as OCR answers. Missing or ambiguous units remain unknown.
4. POST JSON to `answer_url`, copying `request_id` and `image_sha256` exactly:

   Text request (`operation: "text"`):
   ```json
   {"request_id":"COPY_ID","image_sha256":"COPY_HASH","status":"ok","text":"10 mg"}
   ```

   Word-location request (`operation: "words"`):
   ```json
   {"request_id":"COPY_ID","image_sha256":"COPY_HASH","status":"ok","words":[
     {"text":"Dose","left":12,"top":8,"width":42,"height":16,"confidence":95,"line_num":1}
   ]}
   ```
   Supply all clearly visible words in reading order, with a tight rectangle
   around each word. Words on the same line share a positive `line_num`; separate
   lines use different numbers. Confidence is 0–100; uncertain or approximate
   readings must not be labeled high-confidence. Boxes must fit the snippet.
   These boxes can affect legend detection or masking text inside the plot.

   If the snippet cannot be read reliably, use:
   ```json
   {"request_id":"COPY_ID","image_sha256":"COPY_HASH","status":"unreadable"}
   ```
   An actually blank image may use `text: ""` or `words: []` with `status: "ok"`.
   Never use empty answers merely to bypass inspection or finish a queue.
5. Continue checking until the original detection/correction completes. Requests
   arise sequentially as the pipeline progresses. A temporary empty queue does
   not mean detection finished. Review the final labels, axes and overlay with
   the user and identify readings that came from the agent.

Successful submission returns `accepted: true`. Wrong image hashes, malformed
boxes and expired requests are rejected. An identical submission is idempotent;
a conflicting answer cannot overwrite the accepted answer. Identical snippets
with identical OCR settings reuse answers only inside the same job queue.

Each request waits 300 seconds by default. `CHARTOCODE_OCR_WAIT_SECONDS` can set
a value greater than 0 and at most 3600; an AI web subprocess has an overall
one-hour limit. With no answer, the worker stops with an explicit AI OCR timeout
instead of silently exporting as if recognition succeeded. Restart the operation
and actively service requests after a timeout. Late answers are rejected.
An `unreadable` response returns no recognized text, so normal unknown-label /
missing-calibration handling applies. Confirm missing axis values with the user.
An empty or inaccurate Tesseract result is not automatically judged by the AI;
automatic fallback occurs on unavailable/failed Tesseract, not semantic errors.

## Direct command-line workflows and file-only agents

For any v46 CLI, set these environment variables before starting the worker:

```powershell
$env:CHARTOCODE_OCR_MODE='auto-agent'
$env:CHARTOCODE_OCR_DIR='C:\absolute\path\to\this-run\ocr'
# Run the normal v46 command with its image, output directory and other options.
```

On macOS/Linux use `export CHARTOCODE_OCR_MODE=auto-agent` and an absolute
`CHARTOCODE_OCR_DIR`. Use a different queue directory for each image/job and do
not share one queue between simultaneous processes. Web jobs choose their own
directories under the existing temporary job folder automatically.

```text
python src/ocr_bridge_v46.py --directory ABSOLUTE_QUEUE_PATH pending
python src/ocr_bridge_v46.py --directory ABSOLUTE_QUEUE_PATH answer --request-id REQUEST_ID --answer-file ABSOLUTE_ANSWER_JSON_PATH
```

The request JSON files live in `requests/`, PNGs in `snippets/`, and accepted
answers in `answers/`. Prefer the CLI or HTTP submission helper to manually
writing answer files: it validates IDs, bounds and deadlines before publishing.
The same protocol works in Claude Code and Codex when their current tools can
view images and run local commands. The agent's permissions still apply.

`CHARTOCODE_OCR_MODE=agent` forces the handoff even if Tesseract is installed
(useful for explicit AI operation or verification). Without an enabled AI mode,
missing Tesseract never starts a queue. Unset the variables after a direct CLI
session if later commands should use ordinary Tesseract behavior.

Calibration still uses the existing axis parsing and calibration guards. AI text
cannot add support for a number format the parser does not understand. When
manual ranges are needed, values must correspond to the selected plot edges;
use the UI's scale box for ticks inside those edges and confirm log/linear scale.
Preserve series IDs, point coordinates and saved evidence when reviewing labels.
