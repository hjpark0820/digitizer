# AI contributor entry point

Read [README_AI.md](README_AI.md) before installing, running or modifying this application. Read
[README.md](README.md) for the user-facing installation and interaction contract.

Preserve the v46 default runtime, saved image/JSON compatibility, current editor
behavior and unrelated user changes. Explain the exact files/functions changed
and the validation performed. Do not assume files outside this repository exist.

When operating the app, follow [src/AI_WORKFLOW.md](src/AI_WORKFLOW.md). Ask
the region question first, with an OCR question only for manual regions, skipping any already answered:

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

Use `--region-mode agent` for AI regions plus agent OCR. For manual regions,
use `--region-mode manual --ocr-engine tesseract|agent` to match the OCR answer. Plain `run.py` starts manual regions and Tesseract on every launch.
`--agent-workflow` remains an alias for agent regions; `--ai-ocr` remains
Tesseract-first agent fallback. Reject conflicting explicit flags.

After launch, stay active and send heartbeats while monitoring status for
GUI-created batches and servicing region/OCR requests. Do not finish with a
"ready" message while the user is selecting inputs. The GUI can queue images
while an agent is inactive, but displays queued/waiting status and never sends
agent heartbeats. A heartbeat must represent an agent actually available to
answer, not a heartbeat-only background process. The program cannot summon
this conversation or perform unattended AI vision. After every selected task
has a terminal state, report the results and return.

For automatic PNG/JPEG batches, actively send heartbeats and service region/OCR
requests until every authorized item completes or has an explicit failure/review
state. View every actual chart and requested snippet. Never use ground truth,
manifests or saved results as region/OCR answers. Keep detection in the background
so requests can be answered. Read [src/AI_OCR.md](src/AI_OCR.md) for answer schemas.

Give each completed image's `/?batch=BATCH_ID&item=ITEM_ID` link in chat, using
the returned IDs. It opens that exact result in the existing GUI for required
human review and correction. Portable image/JSON pairs remain usable through
the existing Edit saved results controls. Also give the batch's `review_dir`
for **Choose edit folder**: the flat review folder contains matching PNG/JSON
pairs for browsing, editing and saving. Check item `review_error` before claiming
a pair is ready. Respect manual-region choice by
letting the user draw in the existing visible browser canvas.

The program cannot invoke your conversation by itself; the operating agent
must remain active. WebFetch alone cannot draw on a canvas or inspect pixels.
