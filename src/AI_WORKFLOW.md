# Starting and operating an agent-assisted workflow

This guide is for the Codex or Claude agent that is operating the local v46
application. The Python application queues requests and waits for answers; it
does not start an agent, call the current conversation, or install an AI model.
An agent with image-viewing and local HTTP/file tools must be active while
processing requests. A supported host scheduler can resume that agent between
batches; see **Automatic chat wake-up** below. A
successful server launch alone does not complete the user's digitization task.
These instructions govern an agent-operated task. A person can still run
`python run.py` and use the existing manual GUI without an agent or a chat
questionnaire. Missing Tesseract leaves OCR unavailable; manual numeric inputs
and the GUI's calibration warnings remain available.

## Ask in the conversation before starting detection

The operating agent first asks the region question in chat and applies the
answer through launch flags or the local API. AI regions automatically use the
operating agent's vision OCR. Ask the OCR question only for manual regions.
The GUI has no region-mode or OCR selector. Skip questions already answered.

1. **Regions:** "Would you like to select the plot and legend regions yourself,
   or have the AI agent identify them?"

   Include this timing note with the question, before the user chooses:
   "AI selection reduces manual work but usually takes longer than selecting the regions yourself in the GUI."
   Offer **Select regions myself — usually faster** and
   **Let the AI agent select regions — less manual work, usually slower**.

2. **Only for manual regions — OCR:** "Should I use Tesseract, installing it if it is
   missing, or read the text with my AI vision tools?"

Ask the region question first. If the answer is AI regions, use agent vision
for both regions and OCR without another question. If the answer is manual
regions, ask the OCR question unless that choice is already known. Do not add
a path or folder question. After an agent-region launch, the GUI shows an
AI-mode input picker for multiple PNG/JPEG files or a folder. The folder option
explicitly includes subfolders. The user reviews the selected file list and
starts the batch there. Manual-region mode retains the existing single-image
picker and Plot area/Legend controls.

Manual regions can use Tesseract or agent OCR. AI regions always use agent OCR;
do not substitute Tesseract because it happens to be installed. Tesseract setup
applies only when the manual-region route's OCR answer selects it.

The GUI selection and start action establish the automatic input scope. Do not
invent a folder or request a path in chat before showing the picker. If the user
already supplied a folder/files and scope in chat, preserve that instruction and
create the batch through the API instead of asking again or making the user
reselect the same inputs. Use the server's new output directory unless the user
specifies another destination; report the returned output path.
Folder batches accept PNG and JPEG images (`.png`, `.jpg`, `.jpeg`).
For a chat-supplied folder, use `recursive: false` when the agreed scope excludes
subfolders; do not broaden a selected set of files. The GUI folder picker includes
subfolders as its label states. Confirm the actual selected list/count from the
batch snapshot after submission.
Keep input images and existing results intact. Use the server's new batch output
directory or the user's chosen output directory, never a guessed previous job.
For selected files, include `files: ["plot-a.png", "subfolder/plot-b.jpg"]` with
paths relative to the chosen folder; authorize recursion when including subfolders.
Automatic Step 5 correction and Fast detection are separate options; leave them
off unless requested. Region choice does not decide the Colour/B&W detector:
`mode: "auto"` retains the existing pixel-based detector selection.

## Select automatic inputs in the AI-mode GUI

Use **Choose images** for multiple PNG/JPG/JPEG files, or **Choose folder
(includes subfolders)** for images in a directory and its subdirectories.
The GUI shows the selected names/count and reports how many unsupported files
were excluded. **Clear selection** resets that list. Selecting files does not
upload them; **Start digitization** submits the selected images as a batch.
AI-region uploads always use the operating agent's vision OCR, with
no OCR or region-mode choice on this panel.

Selection and queuing remain possible when an agent is inactive, but the GUI
reports queued/waiting status honestly. The GUI never sends an agent heartbeat
and cannot supply AI region/OCR answers. The operating agent must remain active
to process the selected batch. Completed items have a **Review** action opening
the existing result editor; **Download batch results (.zip)** includes results
and portable review pairs.

## Check OCR availability and launch

Install the Python requirements in the repository's local environment. The
`pytesseract` package is a wrapper; it does not install the external Tesseract
program. `GET /agent-workflow/status` reports
`tesseract: {available, version, error}`, the current `settings`, and the agent
connection state. Show a missing executable as missing, never as installed.

If the user selected Tesseract and it is unavailable, use the current official
[Tesseract installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html)
for the user's platform. On Windows, verify the installer source and executable
location; on macOS, verify the available package manager and its package; on
Linux, verify the distribution and its package manager. Inspect the machine
before selecting an installation command. Do not invent or copy an unverified
platform command. Install only after the user has chosen the Tesseract/setup
route; choosing agent OCR does not authorize a Tesseract installation.
Check `tesseract --version` and the running server's availability status after
setup. A successful Python import is insufficient. Restart the server if its
process needs an updated PATH. Report a failed setup honestly and obtain the
user's choice before changing the OCR engine.

Packaged launcher examples:

```text
python run.py --region-mode agent --port 8000
python run.py --region-mode manual --ocr-engine agent --port 8000
python run.py --region-mode manual --ocr-engine tesseract --port 8000
python run.py
```

The first command selects AI regions and automatically enables agent OCR.
The next two commands apply the manual-region route's OCR choice. Plain
`python run.py` starts with manual GUI regions and Tesseract on every launch,
regardless of earlier stored workflow settings. `--region-mode manual|agent`
selects who supplies regions; the user does not repeat that choice in the GUI.
The legacy `--agent-workflow` flag is an alias for agent regions and conflicts
with explicit `--region-mode manual`. In the development workspace use
`python src/unified_server.py` with the same flags and `--host 127.0.0.1`.
`--ocr-engine agent` always uses agent OCR. AI regions cannot use Tesseract or
Tesseract-first fallback; incompatible explicit launch flags are rejected.
For manual regions, `auto-agent` tries Tesseract and falls back only when it is unavailable or its
call fails; it does not detect plausible but inaccurate returned text.
`--ai-ocr` remains a legacy alias for that fallback. Conflicting explicit OCR
flags are errors. Use one explicit engine for a new operation.
For an already running server owned by this task, the agent may apply the
answers with `POST /agent-workflow/settings`, supplying both `region_mode` and
`ocr_mode`, then verify them with `GET /agent-workflow/status`. Do not change an
unrelated user's server or active operation to reuse its port.

Run the server in the environment's supported background process so region and
OCR requests can be answered concurrently with detection. On Windows keep
background helpers hidden. Use a free port and stop only a server you started.
Verify `/openapi.json` and `/agent-workflow/status`, then open or give the exact
local URL. The existing GUI remains the place for human review and correction.
Browser integration is optional for an agent using local image and HTTP tools;
the user still needs a visible browser for manual regions and result editing.
WebFetch alone does not draw on a canvas or read actual image pixels.

## API and active service loop

### Automatic chat wake-up

The digitizer's request/answer protocol is independent of the coding agent.
Codex, Claude Code, or another agent can operate it if the agent can inspect
actual local images and make local HTTP requests. Automatically resuming an
idle conversation is a separate capability of the agent host. Installing or
starting this Python server does not configure that capability.

When the user requests automatic wake-up, use the host's supported scheduling
or session-resume integration and verify registration before claiming it works:

- **Codex desktop:** a native scheduled task attached to the same chat can
  periodically check the queue, for example every minute. Register it through
  the app's automation tool, preserving an existing matching task instead of
  creating duplicates. This is scheduled polling, not an immediate upload
  event. The computer, app, and local digitizer server must remain running.
- **Claude Code or another host:** use that host's documented scheduler or a
  separately configured session runner. The Codex automation does not transfer
  to other hosts. A programmatic agent invocation is not proof that the user's
  existing interactive chat has been awakened. No Claude-specific wake-up
  adapter is bundled with the digitizer.
- **Without a verified wake-up integration:** keep the operating agent active
  while waiting for GUI input and clearly report that closing or ending the
  agent stops servicing requests.

The scheduled prompt must target the authorized local server and verify its
identity using known batch metadata before operating; a reused port alone is
insufficient. First inspect status without sending a heartbeat. If there is no
eligible work, return quietly. Do not retry terminal items or broaden the
user's selected inputs. If another operator has an active heartbeat, defer
rather than answer the same requests concurrently: the current server has no
exclusive agent-claim mechanism.

Once pending work is found, send a real heartbeat and service every region and
OCR request until the submitted batch reaches terminal states. Do not defer
individual OCR answers to later scheduled checks: OCR requests have deadlines.
Report actual completion or required review once; remain quiet on unchanged
checks. Define a stop condition for a closed server and honor the user's stop
request. A configured scheduler must never be represented as an active agent,
and neither the browser nor a heartbeat-only process may impersonate one.

This wake-up route applies to the durable AI-mode file/folder batches. A manual
GUI run using agent OCR still requires an active operator when `/digitize` is
submitted; scheduled polling alone does not change that endpoint's HTTP 409
behavior when no agent is connected.

Host references: [Codex scheduled tasks](https://learn.chatgpt.com/docs/automations?surface=app)
and [Claude Code programmatic operation](https://code.claude.com/docs/en/headless).

### Request processing

All routes below are relative to the running local server. Read its OpenAPI
schema for the complete payload fields and validation limits.

| Route | Purpose |
| --- | --- |
| `GET /agent-workflow/status` | Actual Tesseract availability, agent activity, region/OCR settings and batch summaries |
| `POST /agent-workflow/heartbeat` | `{"agent_id":"YOUR_SESSION_ID"}` announces an actively operating agent |
| `POST /agent-workflow/settings` | Agent applies the chat answers: `region_mode: "manual"` or `"agent"` and `ocr_mode: "tesseract"`, `"agent"` or `"auto-agent"` |
| `POST /agent-workflow/batches` | Start a scoped local folder batch without blocking the agent on detection |
| `POST /agent-workflow/batches/upload` | GUI submits selected PNG/JPEG files as multipart `files` with JSON `options` |
| `GET /agent-workflow/batches/{batch_id}` | Inspect item states, counts, results, `output_dir` and `pending_ocr_count` |
| `GET /agent-workflow/batches/{batch_id}/results.zip` | Download the batch's available results for review |
| `GET /agent-workflow/requests` | Obtain pending region requests with the actual image, hash, dimensions and `answer_url` |
| `GET /ocr/requests` | Obtain actual OCR snippet requests; protocol in [AI_OCR.md](AI_OCR.md) |
| `POST /agent-workflow/batches/{batch_id}/cancel` | Cancel the user's batch |
| `POST /agent-workflow/batches/{batch_id}/retry` | Retry selected `item_ids`; `include_review` controls whether review items are included |

Send a heartbeat about every 30 seconds while working, including while viewing
images or waiting for processing. Agent activity expires after 90 seconds;
accepted region/OCR answers also refresh it. A heartbeat advertises availability,
not correctness or completion. After the chat choice and startup, keep the
agent turn active while the user selects images in the AI-mode GUI unless a
supported automatic wake-up has been registered and verified. Poll status
for GUI-created batches and service region/OCR requests as soon as they appear.
Do not return a final "ready" message and leave the user with an unserviced queue
unless the verified host scheduler will resume processing; disclose its
schedule instead of claiming immediate event delivery.
In manual-region mode with AI OCR, keep sending heartbeats and polling while
the user uploads a chart, draws regions and presses Run in the existing GUI.
Only the input picker and batch start action live in the AI-mode GUI; the region
choice and any manual-mode OCR choice remain in the conversation.
Explicit agent OCR submitted to `/digitize` is rejected with HTTP 409 when no
agent heartbeat is active. Do not claim readiness while leaving that queue unattended.
Keep servicing both queues until the batch itself
has no pending or running items. An empty queue between pipeline stages is normal
and must not end the agent's work. Do not block on `/digitize` before answering
OCR; keep long requests in the background.
Send heartbeats only while an agent is actually available to inspect and answer
requests. A heartbeat-only background process is not an AI worker. The app
cannot continue agent vision unattended or wake this conversation by itself.
After all images in the user's selected batch/task have terminal states, report
the results and return. Future automatic processing requires the separately
verified host scheduler; do not imply that the server alone supplies it.
If the agent stops before completion, explain that it is no longer servicing
requests rather than keeping a false active-agent signal alive.

For each region request:

1. Match the batch, item, image path and SHA-256 hash to the authorized scope.
   Open the **actual image** with an image tool. Use its dimensions to inspect
   the plotting frame and any legend. Do not derive regions from a filename,
   sidecar result, synthetic manifest, reference coordinates or ground truth.
2. Identify the plot interior bounded by the axes and the legend containing the
   relevant swatches and labels. Use inclusive pixel coordinates
   `[x0, y0, x1, y1]` within the supplied image dimensions. The plot region is
   not the whole page or the title/axis-label margins. OCR crops are generated
   from these regions by the existing pipeline.
3. POST to the request's `answer_url` using its exact image hash:

   ```json
   {"image_sha256":"COPY_HASH","status":"ok","plot_area":[100,50,700,500],"legend_box":[520,70,680,160]}
   ```

   The example coordinates show the schema only; measure every real image.
   If no legend is visible, use `"legend_box": null, "no_legend": true`.
   That route assumes one series; do not invent hidden legend entries.
   If the image is unreadable or a region cannot be determined responsibly,
   submit `{"image_sha256":"COPY_HASH","status":"unreadable","reason":"EXPLAIN_VISIBLE_PROBLEM"}`.
   The item needs review; an unreadable image is not a successful extraction.
4. Service the resulting OCR requests using [AI_OCR.md](AI_OCR.md). Open every
   actual snippet, preserve visible signs, decimals, exponents, labels and
   units, and submit word boxes in snippet pixels. Reading the full image does
   not replace inspecting the requested crop. Do not fill ambiguous text from
   earlier results, known synthetic values or expected scientific values.
5. Recheck the batch and continue with the next request. Record failed and
   review items. Resolve a fixable problem and use the retry route as appropriate;
   if user input is necessary, identify the exact image and unresolved choice.
   Process other eligible items while awaiting that input.

Do not claim the folder is finished until every requested item has a terminal
state. Report successful, failed, cancelled and review counts separately with
the output directory. If cancelled or blocked on required input, explain that
remaining images were not completed. A running server or saved JSON alone is
not evidence that all images were processed correctly.

## Example: select automatic inputs in the GUI

> Read AGENTS.md and README_AI.md and set up the app. Use agent-selected regions
> and AI vision OCR. Open the AI-mode file/folder picker; I will choose the PNG/JPEG
> images there. Stay active, view each actual chart and OCR snippet, and keep
> answering requests until all selected images finish or have explicit error/review
> states. Then show the results in the existing editor for my review.

The user has already chosen AI regions, so do not ask an OCR question or
request a path in chat. Start with `--region-mode agent --ocr-engine agent`,
send a heartbeat, open the page, and keep monitoring while the user chooses
files or a folder, reviews the file list and starts processing. The GUI submits
the selected images to `/agent-workflow/batches/upload`; use the returned/status
batch ID to service the same queues described above.

If the user already specified a local folder in chat, the direct API alternative
remains available. For example, an instruction to process all PNG/JPEG charts
directly in `C:/charts/pk-study` without subfolders can use this batch body:

```json
{
  "folder": "C:/charts/pk-study",
  "recursive": false,
  "region_mode": "agent",
  "ocr_mode": "agent",
  "mode": "auto",
  "correct": false,
  "fast_detection": false
}
```

Save the returned batch ID and use the service loop above until all its items
are terminal. The batch endpoint starts work; it does not read the images for
the agent. Provide each completed image's explicit review link in chat:
`http://127.0.0.1:PORT/?batch=BATCH_ID&item=ITEM_ID`, using IDs from the returned
batch snapshot. The link opens that exact result in the existing GUI; it does
not select the newest job.

For a selected subset, add `"files": ["plot-a.png", "plot-b.jpg"]` to the
batch body, using the actual enumerated filenames. Use that list as well when
the user limits the scope further, for example to PNG files only.
Batch outputs are kept in a distinct `batch_ID` directory under the
output folder; use the returned `output_dir` rather than assuming a path.
Item states include `pending_regions`, `ready`, `waiting_agent`, `running`,
`completed`, `failed`, `needs_review` and `cancelled`. Only the last four are
terminal; `needs_review` explicitly records incomplete or uncertain work.

## Human review and saved results

Automation prepares results for human review; it does not approve extracted
measurements. Give completed per-image links using `/?batch=BATCH_ID&item=ITEM_ID`.
The ordinary **Edit saved results** controls can also open portable image/JSON
pairs. The batch snapshot's `review_dir` identifies a flat `review/` folder
containing matching `<itemID>.png` and `<itemID>.json` files for completed
results. Give that path to the user, who can open it with the existing
**Edit saved results → Choose edit folder**, browse the pairs, edit and save.
Retries use a job suffix to preserve earlier pairs; the batch ZIP also includes
`review/`. Item fields `review_image_path`, `review_json_path` and
`review_job_id` identify the saved pair; report `review_error` if pair creation
failed instead of claiming a usable pair exists. Keep the per-image link as
the direct route to that item's result.

Ask the user to review each chart's plot and legend, series labels,
overlay, reconstructed plot, axis scale/calibration, units and coordinate table
in the existing GUI. Human review and correction are required before relying on
the data. The editor supports point additions, deletions and moves, recalibration,
and supported Step 5 correction. Save the corrected image-linked JSON and keep
the original image alongside it; Excel/CSV exports do not replace that evidence.

Units must come from readable axis/title text, with missing or ambiguous units
left unknown. Preserve printed log units rather than silently exponentiating.
Failed calibration remains explicitly labelled in pixel units until corrected.
Recognized units appear in the GUI coordinate-table headings, such as `X (h)`.
An unrecognized unit leaves the calibrated heading as `X (axis units)`; it does
not supply a unit by assumption. CSV/Excel exports retain separate unit and
unit-status columns. Pixel or normalized coordinates never receive physical-unit
labels, even when title text was read. Unit recognition does not convert numeric
values. Manual numeric limits skip tick OCR; axis-title/unit OCR remains separate
and may still request agent input.
Do not reinterpret generated OCR or automated extraction as a manual review.
