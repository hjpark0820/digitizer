# Plot Digitizer v46

A local web app for extracting data points from chart images, editing them in a browser, and saving the results as JSON, Excel, or CSV. Supports colour and black-and-white charts. AI agents maintaining the code should read [README_AI.md](README_AI.md).

## Installation and startup

The verified environment is **Windows, Python 3.12, and CPU execution**. Open the app at `localhost` in Chrome or Edge to access folders and save JSON files alongside their original images.

**This is a standalone v46 distribution. `run_A4_auto_v45.py` is neither required nor included.**

Use this folder as the root of your GitHub repository. The original development project's `data`, `models`, `experiments`, and `outputs` folders are not required.

In Windows PowerShell, navigate to this folder and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run.py
```

Open **http://localhost:8000** in your browser. Keep the terminal open while using the app, and press `Ctrl+C` to stop the server. You do not need to activate the virtual environment.

If another program is using port 8000, run the following command and open `http://localhost:8001`:

```powershell
.\.venv\Scripts\python.exe run.py --port 8001
```

The equivalent macOS/Linux commands are `python3.12 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, and `.venv/bin/python run.py`. Installation of this distribution was verified on Windows.

### OCR and GPU support

- **OCR (optical character recognition)**: To read axis numbers and legend text automatically, install the Tesseract executable separately and add it to `PATH`. Check the installation with `tesseract --version`. Installing the `pytesseract` Python package does not install the Tesseract executable. If OCR is unavailable or a reading is uncertain, enter the axis values manually and review the results.
- **CPU-only operation**: Current v46 detection does not require trained model weights, `timm`, or YOLO.
- **Optional GPU acceleration**: If a CUDA-capable PyTorch installation is already available, B&W v46 checks the GPU when selecting a compute device automatically. Otherwise, it uses the CPU. PyTorch is excluded from the default requirements. To select the CPU explicitly, set `$env:BW_V46_REFINEMENT_BACKEND='cpu'` in PowerShell before starting the app. The standalone environment checks for this distribution did not use a GPU.

## Using the app with Codex or Claude Code

You can ask an AI agent to download the repository locally, prepare the virtual
environment and packages according to `README_AI.md`, and start the server.
Open the resulting URL and draw the regions yourself using the existing
**Plot area** and **Legend** buttons.

- **Codex desktop with a built-in browser available**: Open the local page in
  the app and select the regions there. In a CLI or an environment without a
  browser integration, open the page in regular Chrome or Edge.
- **Claude Code**: Use its connected browser when Chrome integration is ready.
  Otherwise, open the local URL manually in your browser.
- **WebFetch alone cannot collect region selections**: It retrieves page
  content and does not provide an interactive canvas for mouse dragging.
  A browser is required.
- **When Tesseract is unavailable**: If the agent starts the app with
  `python run.py --ai-ocr`, the program saves the actual OCR image snippets
  and waits. Detection continues when Codex or Claude reads the snippets and
  submits answers. Have the agent keep handling requests according to the
  [AI OCR handoff guide](src/AI_OCR.md). Image-viewing and local command tools
  are required; no separate API key is needed. If the agent stops or does not
  answer, the request reports an error after five minutes by default. Confirm
  unclear text instead of guessing. Ordinary execution uses Tesseract or
  manual input as before.

After selecting the regions, press **Run** to send their coordinates to the
detection server. Selecting a region alone does not send it to the AI
conversation. To delegate follow-up work, save the results and give the agent
the JSON path, or let the agent inspect the results through a connected browser.

## Extracting data from a new chart

1. Select a PNG or JPEG using **New detection: choose image**.
2. Choose **Colour / B&W** mode. If needed, adjust the image resolution or rotation under **Image preparation**.
3. Select the plotting region with **Plot area**. If the chart has a legend, select it with **Legend**. Running without a legend assumes **one data series**.
4. Enter the actual axis values. Enable the corresponding option for logarithmic axes. Results without sufficient axis information may use pixel coordinates instead of physical values.
5. For colour charts, **Colour chart series** can usually remain set to `Auto`. If needed, explicitly select a chart with markers or a line-only chart.
6. Press **Run**, then review the points overlaid on the original image and the reconstructed plot.
7. Supported results can be corrected automatically with **Step 5**. Older JSON files without saved correction evidence may support manual editing only.

When automatic detection is uncertain, the app may display a failure or a reason for withholding a result. Check the selected regions and mode. The reconstructed plot is a review view that connects the current points with straight lines in x-coordinate order.

## Opening saved images and JSON files

1. Select an image folder using **Edit saved results → Choose edit folder**.
2. Only image/JSON pairs with matching names, such as `plot.png` and `plot.json`, appear in the editing list.
3. Select a file from the list or use `<` / `>` to update the **Original / Overlay / Reconstructed plot** previews automatically.
4. Press **Open selected image for editing**. If several versions are available, select the saved version to edit.

Select the folder again after refreshing or reopening the page. In browsers without folder access support, use **Legacy import — select image + JSON manually** to load individual files. Direct saving to a folder requires Chrome or Edge's folder access feature.

## Dedicated editing view

Pressing Edit switches from the previous workspace to a dedicated editing view. File navigation, series selection, zoom, and save controls remain available within the view.

| Control or gesture | Action |
| --- | --- |
| Colour/series button | Show only the selected series in the editing image and reconstructed plot |
| All symbols | Show every series |
| Drag a point in Edit points mode | Change the point's coordinates |
| Click empty space in Edit points mode | Add a point to the selected series |
| Left-drag on empty space | Pan the zoomed image |
| Select Pan, then left-drag | Pan even when the drag starts on a point |
| Select a point, then Delete / Remove selected | Delete the point |
| `+` / `−` / Ctrl+mouse wheel | Zoom in or out; the former 600% limit has been removed |
| Fit | Show the entire image |
| `<` / `>` | Move to the previous or next editable file in the same folder |
| Coordinates table | Expand or collapse the coordinate table |
| Back / Resume editing | Switch views within the current page while preserving edits, zoom, and scroll position |

`Back` does not save changes to a file. Press **Save result versions (.json)** before refreshing the page or closing the browser.

## Saved formats

- **JSON**: Stores result history, edited coordinates, axis information, and available correction evidence in a JSON file with the same name as the original image, alongside that image. Initial detection, Step 5, and manual edits are stored as separate versions. Keep the original image and JSON together. Rotated or resized working images may be embedded in the JSON, increasing its size.
- **Excel / CSV**: Coordinate tables for analysis. They do not replace the complete correction evidence needed to resume editing.
- Saving and exporting preserve all series, even when a colour button filters the view to one series.
- The app's JSON limit is 128 MiB. Editing data larger than 1 MiB is transmitted as a file upload.

The server creates temporary results under `unified_digitizer_jobs` in the operating system's temporary folder. Temporary results do not replace the JSON files you save. Image processing runs on the local Python server.

## Folder layout

```text
.
├── README.md              # Installation and usage guide for people
├── README_AI.md           # Architecture, maintenance, and verification guide for AI agents
├── AGENTS.md              # Instructions entry point for Codex and other AI agents
├── CLAUDE.md              # Imports the instruction documents for Claude Code
├── requirements.txt       # Verified Python dependencies
├── run.py                 # Recommended launcher
├── RUNTIME_MANIFEST.json  # Included sources, reference locations, and SHA-256 hashes
└── src/
    ├── unified_server.py
    ├── index.html
    ├── image_sidecar_v46.js
    ├── run_A4_auto_v46.py
    ├── run_A4_color_v46.py
    ├── color_pipeline_v46.py  # v46 processing stages
    ├── chart_analysis_v46.py  # Shared analysis functions
    ├── legend_palette_v46.py  # Legend and palette construction
    ├── analysis_session_v46.py # Per-image analysis state
    ├── legend_optional_v46/
    ├── type3_v46/
    └── ...                # Runtime modules for detection, correction, and saving
```

The runtime files in `src` were copied from the development workspace. Legacy Vision Transformer (ViT) detection CLIs, training code, experimental results, and model files are excluded from this distribution. You can also run `src/unified_server.py` directly, but `run.py` locates the app independently of the current working directory and defaults to `127.0.0.1`.

## Troubleshooting

- **Address already in use**: Stop the existing server or use `run.py --port 8001`.
- **New features do not appear**: Save your changes and refresh the page. If you changed Python server code, restart the server as well.
- **A save error mentions `1024KB` or an older format**: Check whether an older server started from another folder is using the same port, then launch `run.py` from this distribution folder.
- **Image/JSON pair mismatch**: Both the filenames and the original image content must match. Do not rename a JSON file to associate it with a different image.
- **Folder selection or save permission problems**: Open `http://localhost:PORT` in Chrome or Edge, replacing `PORT` with the server's port number, and select the folder again.

## Distribution verification

On 2026-09-21, the distribution was checked in a temporary location separate from the original project, using a fresh Python 3.12 virtual environment.

- Installation from `requirements.txt` and package dependency checks passed.
- In a distribution without v45 files, hashes for 143 runtime files were verified and 141 Python modules imported successfully.
- Five detection cases and their respective Step 5 corrections passed from an independent folder. Coordinates, colours, labels, and axis information matched the previous version.
- Server startup, HTML/JavaScript delivery, image resizing, and rotation were verified.
- Generated samples passed Colour/B&W marker detection with and without legends, line-only detection without a legend, and Step 5 correction from each result's saved evidence.
- Uploading a 1,472,804-byte JSON file and saving, reloading, and exporting 55,000 points to Excel passed.
- Automatic compute-device selection fell back to the CPU when PyTorch was unavailable.

Verification images, virtual environments, and result files are not included in this distribution. These sample runs check installation and feature integration; they do not establish detection accuracy for every real-world chart.

When uploading to GitHub, place the **contents of this folder at the root of the new repository**. You do not need to include virtual environments, personal images, saved JSON files, or temporary results.

## Removing the v45 dependency

The server explicitly selects `run_A4_auto_v46.py`. Reading and rewriting v45 source for execution, and automatic fallback to legacy detectors, have been removed. If a marker legend cannot be analyzed, the app reports an error; check the selected region or choose `Auto` / `line-only` mode. If the `PLOT_PIPELINE` environment variable points to an old v45 file, unset it or change it to the v46 path. Restart the existing server to apply the changes.
