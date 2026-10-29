# Claude Code project instructions

@AGENTS.md

Read @README_AI.md before installing, running or modifying this application.
Read @README.md for the user-facing setup and interaction contract.

When asked to operate the app, follow src/AI_WORKFLOW.md and ask the region question first
in the conversation, with OCR choice only for manual regions, skipping answers already supplied:

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

Apply the choices through
`--region-mode agent` (automatically agent OCR), or manual regions with
`--ocr-engine tesseract|agent`. You can also use the settings
API for the server owned by this task. Plain `run.py` starts manual regions and
Tesseract every time. Install Tesseract only when the user chose that route,
then verify the executable. `--agent-workflow` and `--ai-ocr` remain legacy
aliases for agent regions and Tesseract-first fallback respectively.

Prepare the local Python environment and requirements, then start the server
in the background.
After launch, stay active and send heartbeats while monitoring status for
GUI-created batches and servicing region/OCR requests. Do not finish with a
"ready" message while the user is selecting inputs. The GUI can queue images
while an agent is inactive, but displays queued/waiting status and never sends
agent heartbeats. A heartbeat must represent an agent actually available to
answer, not a heartbeat-only background process. The program cannot summon
this conversation or perform unattended AI vision. After every selected task
has a terminal state, report the results and return.
View each actual chart and every OCR snippet; never use ground truth, manifests
or saved results as answers. Keep working until every selected image completes
or has an explicit failure/review state. Follow src/AI_OCR.md for answer schemas;
do not block on long detection requests before answering the waiting worker.

Give completed per-image `/?batch=BATCH_ID&item=ITEM_ID` links in chat for
required human review/correction in the existing GUI. Also give the batch's
`review_dir` for **Edit saved results → Choose edit
folder** to browse, edit and save its matching PNG/JSON pairs. Check item
`review_error` before claiming a pair is ready. Use connected browser
tools when available; WebFetch cannot draw on the canvas or inspect pixels.
The program cannot invoke your conversation by itself.

Preserve v46-only runtime behavior, saved-session formats and unrelated changes.
