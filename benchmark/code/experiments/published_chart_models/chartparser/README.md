# ChartParser official released-head reproduction

This experiment applies the checkpoint linked by [ywking/ChartParser](https://github.com/ywking/ChartParser) to the shared 178 BW native panel crops. Production `src/` is untouched. Original images are never uploaded. No training or threshold fitting uses these inputs.

The follow-up colour experiment uses a separate `inputs_color/manifest.json` and `results_color/` destination. It includes all 274 prior v46-eligible colour panels plus 40 reviewed no-legend line panels, for 314 panels. The neural heads do not require a legend; v46's missing-legend status remains explicit in the comparison instead of dropping those cases. The same checkpoint, preprocessing and thresholds are used, without colour-specific retuning. BW results are not overwritten.

## What was reproduced

- Exact upstream commit: `1ce7ee206b4b97fefdce265fbab22de6373988a7`.
- Checkpoint: official README Google Drive file `1P8i5cR8pr9ncKjaIzksrnViRZje6zf6C`, `epoch-19.pth`, 949,941,190 bytes.
- SHA256: `06f8cd58e9a46e1a66a13bb6c982671ade3b903eec6c430adfc45378ebdeba87`.
- The wrapper loads with `torch.load(weights_only=True)`, never unrestricted pickle.
- Three network heads: untyped keypoint heatmap+offsets, two-class curve segmentation, eight-class chart-element boxes.
- Upstream `single_img_demo`: OpenCV BGR input stretched to 512x512, ImageNet mean/std applied in that BGR order; last-stack keypoint sigmoid, 3x3 local maximum suppression, score > 0.3; detector confidence 0.4 and NMS IoU 0.45.
- Curve segmentation is the argmax of two output channels, not series-specific paths. Native masks are nearest-neighbor-expanded 512 outputs, **not native-resolution inference**.
- Keypoints are truncated to integer 512-space coordinates as in the demo, then returned to the native crop. Additional subpixel coordinates are exported separately.

## Necessary compatibility changes and limits

The upstream demo imports nonexistent `tools2` modules and has commented-out OCR and series-grouping calls. We import the existing `tools.decode_heatmap` directly and do not execute the demo, delete its result directory, or call its Azure OCR code. **These results are released-head inference, not full end-to-end physical-value/series digitization.** Series reconstruction is deliberately absent rather than invented.

The checkpoint has six detector anchors per scale, but current source declares three. `run_heads.py` reconstructs only that head dimension from the checkpoint's own `model.24.anchor_grid` buffer, then strictly loads **all 1,939 state-dict entries**. No tensor is dropped, no head remains randomly initialized. The resulting network has 79,016,345 parameters. Exact before/after anchor values are in `results/provenance.json`.

The README title names *Efficient extraction of experimental data from line charts using advanced machine learning techniques*, while its citation names *AI-ChartParser* (CGF 2025). The reproducible identity here is the explicit repository commit plus official checkpoint. No claim is made that missing paper-stage code was reproduced.

The pinned repository contains no LICENSE or the mentioned requirements.txt. Keep these sources and weights as an isolated local research checkout; no production merge or redistribution is performed.

## Results and coordinate convention

`results/<panel_id>/result.json` contains:

- `raw_points`: all model keypoint predictions, including legend/text/outside-plot proposals.
- `points`: the same predictions restricted to the already-reviewed shared plot ROI, excluding the shared legend ROI. This is scope filtering, not new neural inference or accuracy filtering.
- `boxes`: all predicted chart-element boxes. Class 3 is absent from upstream demo's label map, so it is explicitly called `unmapped_class_3`.
- `curve_mask.png`: full-crop binary segmentation; display code should scope the overlay to the same plot ROI.
- `raw_heads.npz`: sigmoid heatmap, regression offsets, two segmentation channels, decoded points, and NMS boxes.

All coordinates are `input_crop_pixels`, x rightward and y downward. Add the shared manifest's `crop_box_upright` top-left offset to obtain upright-original-image coordinates. The manifest records any original source rotation. No axis-value conversion is claimed.

Timing fields separate synchronized GPU network-forward time and per-image processing including decoding and native-mask export (excluding model loading and final JSON writing). They are **not comparable as full-pipeline timing** to native-resolution v46, which also assigns series and runs correction.

## Run and audit

Use the existing Python 3.12 / PyTorch 2.11 CUDA runtime. Only missing `yacs==0.1.8` and `prefetch-generator==1.0.3` are installed under `python_deps`, leaving global Python unchanged.

```powershell
python experiments/published_chart_models/chartparser/run_heads.py --smoke
python experiments/published_chart_models/chartparser/run_heads.py
python experiments/published_chart_models/chartparser/audit_results.py
python experiments/published_chart_models/chartparser/run_heads.py --manifest experiments/published_chart_models/inputs_color/manifest.json --output experiments/published_chart_models/chartparser/results_color
python experiments/published_chart_models/chartparser/audit_results.py --manifest experiments/published_chart_models/inputs_color/manifest.json --output experiments/published_chart_models/chartparser/results_color
```

`results/audit.json` verifies all input hashes, coordinate transforms, ROI membership, binary masks, finite raw outputs, unchanged upstream Python source, and bitwise preprocessing equivalence to the demo transform. This audit is an implementation/data-integrity check, **not a ground-truth accuracy evaluation**.
