# AI contributor entry point

Read [README_AI.md](README_AI.md) before installing, running or modifying this application. Read
[README.md](README.md) for the user-facing installation and interaction contract.

Preserve the v46 default runtime, saved image/JSON compatibility, current editor
behavior and unrelated user changes. Explain the exact files/functions changed
and the validation performed. Do not assume files outside this repository exist.

When operating the app for a user, follow the agent-assisted OCR and browser
handoff instructions in README_AI.md. Use image reading when Tesseract is
unavailable and the current tools support it. Let the user draw plot/legend
regions in a visible browser; WebFetch alone cannot collect those selections.

For AI operation, use `run.py --ai-ocr` and actively handle `/ocr/requests` as
documented in [src/AI_OCR.md](src/AI_OCR.md). View each actual snippet and submit
the JSON answer so the waiting worker continues. Keep long-running detection
in the background; do not wait for its HTTP response before answering OCR.
The app cannot invoke your conversation by itself.
