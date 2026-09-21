# Claude Code project instructions

@AGENTS.md

Read @README_AI.md before installing, running or modifying this application.
Read @README.md for the user-facing setup and interaction contract.

When asked to operate the app, prepare the local Python environment, install the
requirements, start the server and open or provide its local browser URL. Let
the user select plot/legend regions in the existing UI. Use connected browser
tools when available; WebFetch cannot render the region-selection canvas.

If Tesseract is unavailable, use your image-reading capability for axis values
and legend text when supported. Follow the calibration/uncertainty instructions
in README_AI.md and service the real local OCR handoff in src/AI_OCR.md.

Preserve v46-only runtime behavior, saved-session formats and unrelated changes.

For AI operation, use `run.py --ai-ocr` and actively handle `/ocr/requests` as
documented in [src/AI_OCR.md](src/AI_OCR.md). View each actual snippet and submit
the JSON answer so the waiting worker continues. Keep long-running detection
in the background; do not wait for its HTTP response before answering OCR.
The app cannot invoke your conversation by itself.
