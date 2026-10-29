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

A plain launch uses manual GUI regions and Tesseract, regardless of earlier workflow settings.

You can use the manual GUI without an agent. If Tesseract is missing, OCR is
unavailable; manual numeric input and calibration warnings remain available.

Open **http://localhost:8000** in your browser. Keep the terminal open while using the app, and press `Ctrl+C` to stop the server. You do not need to activate the virtual environment.

If another program is using port 8000, run the following command and open `http://localhost:8001`:

```powershell
.\.venv\Scripts\python.exe run.py --port 8001
```

The equivalent macOS/Linux commands are `python3.12 -m venv .venv`, `.venv/bin/python -m pip install -r requirements.txt`, and `.venv/bin/python run.py`. Installation of this distribution was verified on Windows.

### OCR and GPU support

- **OCR (optical character recognition)**: Choose Tesseract or the operating agent's image reading. Tesseract is an external program installed separately from the `pytesseract` Python wrapper. If you choose Tesseract, follow the [official platform installation guide](https://tesseract-ocr.github.io/tessdoc/Installation.html), verify `tesseract --version`, and verify that the selected environment can use it. During agent operation, the agent also checks the local server's OCR status. AI OCR requires an active agent with image tools. If a reading is uncertain, review it and enter manual values as needed.
- **CPU-only operation**: Current v46 detection does not require trained model weights, `timm`, or YOLO.
- **Optional GPU acceleration**: If a CUDA-capable PyTorch installation is already available, B&W v46 checks the GPU when selecting a compute device automatically. Otherwise, it uses the CPU. PyTorch is excluded from the default requirements. To select the CPU explicitly, set `$env:BW_V46_REFINEMENT_BACKEND='cpu'` in PowerShell before starting the app. The standalone environment checks for this distribution did not use a GPU.

## Using the app with Codex or Claude Code

Ask your agent to read `AGENTS.md` and `README_AI.md`, prepare the local Python
environment, and operate the app. The agent first asks about regions **in your chat**, then asks about OCR only
if you choose manual regions,
skipping any choices you already supplied:

1. **Regions:** "Would you like to select the plot and legend regions yourself,
   or have the AI agent identify them?"

   Include this timing note with the question, before the user chooses:
   "AI selection reduces manual work but usually takes longer than selecting the regions yourself in the GUI."
   Offer **Select regions myself — usually faster** and
   **Let the AI agent select regions — less manual work, usually slower**.

2. Only for manual regions: "Should I use Tesseract, installing it if it is missing, or read the text
   with my AI vision tools?"

AI regions automatically use the agent's vision OCR, without an OCR question.
There is no path question. After you choose AI regions, use the
AI-mode GUI to select the inputs:

1. **Choose images** selects multiple PNG/JPG/JPEG files. **Choose folder
   (includes subfolders)** selects images in a folder and its subfolders.
2. Check the selected file names/count; unsupported files are excluded and
   counted. **Clear selection** resets the list.
3. Press **Start digitization** to upload the selection and create the batch.
   Choosing files alone does not upload them.

Region mode and any manual-mode OCR choice are made in chat; the GUI has no
selectors for them. Manual-region mode keeps the existing single-image picker and drawing
controls. Tesseract is installed only if you choose that route. If you already
provided an input folder/files and scope in chat, the agent can use the batch
API directly without asking again or making you reselect those inputs.

For example, this request already selects the complete AI route:

> Read AGENTS.md and README_AI.md and set up the app. Use agent-selected regions
> and AI vision OCR. Open the AI-mode file/folder picker; I will choose my images
> there. Stay active and answer requests until all selected images finish or
> have an explicit error/review state, then let me review them in the editor.

The corresponding launch is
`python run.py --region-mode agent --ocr-engine agent`.
Use `--region-mode manual` for drawing regions yourself and
`--ocr-engine tesseract` for Tesseract only in manual-region mode. AI regions
automatically use agent OCR; explicit Tesseract/fallback flags with AI regions
are rejected. Plain `run.py` always starts manual
regions and Tesseract. `--agent-workflow` remains an alias for agent regions;
`--ai-ocr` remains Tesseract-first fallback (`--ocr-engine auto-agent`).
Conflicting explicit flags are rejected.

The agent stays active while you select images and press Start, monitors the
created batch, and sends heartbeats only while it can answer requests. The GUI
allows selection/queuing when the agent is inactive, but honestly displays
queued/waiting status; it cannot run AI vision by itself. After all selected
tasks have terminal states, the agent reports results and returns.

The agent checks actual engine availability through the local API, views each
actual chart and OCR snippet, and keeps servicing requests until the requested
scope finishes or has explicit error/review states. Original inputs and saved
results remain intact. The server cannot summon the agent; if the agent stops,
work can wait or time out. No separate AI API key is required. See the
[agent workflow guide](src/AI_WORKFLOW.md) and
[OCR answer guide](src/AI_OCR.md) for operation details.

For automatic results the agent provides a separate local link for each image,
using `/?batch=BATCH_ID&item=ITEM_ID`; the GUI's **Review** action opens the
same result. **Download batch results (.zip)** downloads available outputs.
Opening the per-image link displays that exact
result in the existing GUI. Portable image/JSON pairs still open through
**Edit saved results**. The agent also gives the batch's `review_dir`: a flat
`review/` folder with matching PNG/JSON pairs. Select that folder using
**Edit saved results → Choose edit folder** to browse, edit and save the results.
Retries preserve earlier pairs, and the batch ZIP includes the review folder.
For manual regions use the existing **Plot area** and
**Legend** buttons, then **Run**. WebFetch alone cannot draw on the canvas or
inspect image pixels; use an available connected browser or ordinary Chrome/Edge.

**Human review and correction are required in the existing GUI.** Check each
chart's regions, labels, units, axes, overlay and extracted coordinates before
using measurements. Use the existing point editor and supported Step 5
correction, then save image-linked JSON. Automation does not certify accuracy.

## Extracting data from a new chart

The following steps describe manual GUI region selection. For agent-selected
regions or a folder batch, use the workflow above and review the same results
and editing controls described below.

1. Select a PNG or JPEG using **New detection: choose image**.
2. Leave **Automatic** selected (the default), or choose **Colour / B&W** to override it. If needed, adjust the image resolution or rotation under **Image preparation**.
3. Select the plotting region with **Plot area**. If the chart has a legend, select it with **Legend**. Running without a legend assumes **one data series**.
4. Leave the X/Y limits blank to read numeric tick labels automatically. The app independently chooses a linear or logarithmic scale for each axis. Axis fields are always visible: enter both limits to override an axis manually, and enable its logarithmic option when appropriate. Manual limits refer to the selected plot edges. To calibrate from interior ticks instead, enable **Use a separate scale box (optional)**, draw that box and enter all four values. Uncertain automatic readings return explicitly labelled pixel coordinates with a calibration warning, not a guessed 0–1 data range.
5. Automatic inspects the selected legend and plot pixels and chooses the Colour or B&W detector. A plot area is required. Results show the selected detector and reason. Colour charts then automatically infer whether the plot uses markers or lines.
6. Optionally enable **Fast detection** before Run. It limits the detection image to 1,400 pixels on its longest side without changing your upload or selected boxes. Axis labels are read from the original image. Small markers may lose detail; leave the option off for full-resolution detection.
7. Press **Run**, then review the overlay, reconstructed plot and **Extracted data — current result** spreadsheet directly on the results page. Opening Edit is not required to view coordinates. The table refreshes after detection and Step 5, shows active points only, and labels uncalibrated axes explicitly.
8. Supported results can be corrected automatically with **Step 5**. Older JSON files without saved correction evidence may support manual editing only.

When automatic detection is uncertain, the app may display a failure or a reason for withholding a result. Check the selected regions and mode. Edit, Step 5, and legend-analysis controls remain visible after Run. If valid series data were saved, you can add or adjust points manually even when automatic detection is incomplete, then save JSON or Excel. Step 5 stays disabled with an explanation until a completed detection provides usable correction evidence. The legend-analysis panel can display the saved series report even when no diagnostic image was generated. The reconstructed plot is a review view that connects the current points with straight lines in x-coordinate order.

Large charts with several similar-colour series can take several minutes. The Run indicator shows elapsed time. Ordinary detection allows up to 15 minutes; a timeout is shown explicitly and the complete process output is saved as `pipeline.log` inside the temporary job folder. **Fast detection** is off by default and affects only the next Run. Images already below the limit are not resized. Results, editing and Step 5 use the same reduced working image and its pixel coordinates; calibrated data values retain their axis units. Saved JSON retains the working image and resolution metadata. Changing the checkbox does not resize an existing result or Step 5 session. Alternatively, **Image preparation** changes the displayed image explicitly and requires redrawing the plot and legend regions.

Automatic is a lightweight routing heuristic, not a guarantee of the best detector. Select the legend when available: black markers with coloured connectors can be routed to B&W when the legend marker bodies are clear. Without a usable legend, connector colour can influence the fallback. Use the manual buttons for ambiguous charts. Step 5 and saved sessions retain the detector that produced the result, independently of the button selected for the next run.

Automatic **axis calibration** is separate from the **Automatic / Colour / B&W** detector choice and works in all three modes. It requires at least three consistent numeric tick readings per inferred axis. You can supply one limit and let the app infer the missing limit when the tick evidence is reliable. Both manual limits skip numeric-tick reading for that axis; an unchecked logarithmic option then means linear. Axis-title/unit OCR remains independent and may still request an agent answer. The result and coordinate table identify automatic, manual, mixed or uncalibrated axes. Step 5 preserves the saved calibration.

Review calibration before using exported numbers. Small text, missing decimals or exponents, overlapping axis titles, decreasing scales, and categorical or multiple-row labels can require manual input. A plot labelled with **log concentration** is read in its printed log units; the reader does not automatically exponentiate those units to concentration. Fitted plot-edge values may extend beyond the outermost printed ticks when the selected rectangle includes margins. Tesseract or the documented AI OCR bridge is needed for automatic reading; manual values remain available without OCR.

Recognized axis-title units appear in the coordinate-table headings, for example
`X (h)` or `Y (ng/mL)`. Unknown units remain labelled `axis units`, while
pixel/normalized coordinates stay explicitly uncalibrated and receive no
physical-unit label. CSV/Excel exports retain separate unit and unit-status
columns. Unit recognition does not convert the numeric values; review labels
and units along with the calibration before using exported measurements.

## Synthetic examples

[examples/](examples/) contains 40 selected synthetic plots in one flat folder: 10 PK B&W, 10 PK colour, 10 ER B&W and 10 ER colour. Filename prefixes identify each category. The [example guide](examples/README.md) and [manifest](examples/examples_manifest.json) provide the tested plot/legend regions, calibration, scores before and after automatic correction, and timing. These are curated demonstration cases, not a representative benchmark or clinical measurements.

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
    ├── AI_WORKFLOW.md      # Agent startup, regions, OCR and folder batches
    ├── AI_OCR.md           # Exact snippet/answer protocol
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
