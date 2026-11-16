# LineEX local reproduction

This experiment runs **the official LineEX L+D keypoint module**, not an invented
approximation and not the complete chart-to-data pipeline. The network gets only
the shared input image. v46 detections, legend swatches, and plot boxes are not
network inputs. Plot/legend boxes are used only to label output points for the
comparison gallery.

## Official assets

- Repository: https://github.com/Shiva-sankaran/LineEX
- Source commit: `b5776d501696087894613fc7accf2180f98f89fe`
- License: Apache-2.0, retained in `../vendor/LineEX/LICENCE.md`
- Checkpoint: authors' `ckpt_L+D.t7`, epoch 150
- Public folder: https://drive.google.com/drive/folders/15bKREf2EBORHBZx_xz53eVJzs9e4cZ7Q
- File: https://drive.google.com/file/d/1QyPykDBusW9LW0YejuTJfgfPr4qICtq8/view
- Size: 487,265,684 bytes
- SHA256: `fb7f1ed279e30e4916f2c02e14ebf2457a3bf4cdb40452e05c8d67702c3023a6`

The checkpoint was inspected with PyTorch's restricted tensor loading. There
were no unsafe pickle globals, and all 503 model state keys matched strictly.
There are 40,562,834 parameters. The original source files were not edited.

## Exact scope and limitations

The official XCiT-small-12-p16 encoder plus six-layer transformer decoder predicts
64 normalized point coordinates. The official inference function removes
background proposals using correlations between nine neighboring color
histograms. It does **not** produce a point confidence score, marker shape, or
series label. Repeated or nearby predictions are not silently deduplicated.

The input crop is read at native resolution, but the official network internally
warps it to **width 288 by height 384**, BGR order, divided by 255. No RGB conversion
or ImageNet mean/std normalization was added. Predictions are truncated and scaled
back to native input-crop pixels. This is not a native-resolution neural model.

LineEX keypoints describe line-chart geometry; a predicted point along a smooth
curve need not be a literal experimental marker center. Therefore point counts
must not be reported as marker precision/recall without appropriate annotations.
The 64-query limit also cannot recover more than 64 distinct coordinates per crop.

The complete published repository additionally contains chart-element detection,
OCR, legend grouping and numeric axis conversion. Those modules are **not run in
this experiment**, so no fabricated reconstruction or series assignments are
exported. In the repository's `pipeline.py`, learned legend-match confidence is
overwritten by `hist_cmp()`'s color-histogram similarity before grouping. That
provided pipeline is especially unsuitable as a faithful shape-discrimination
baseline for same-color black-and-white series; the pure keypoint result remains
available without conflating this separate issue.

## Compatibility-only changes

`run_keypoints.py` calls the same upstream registered XCiT constructor directly
because timm 1.x adds unsupported `pretrained_cfg` arguments to the older model.
None-valued factory arguments are stripped as timm 0.5.4 did. This changes neither
layers nor trained weights. Python integer coordinates preserve NumPy 1.x signed
slice arithmetic in the unmodified background gate; NumPy 2.x unsigned scalar
underflow would otherwise change edge behavior. Model loading uses
`weights_only=True`, `strict=True`; evaluation uses inference mode. No global
package installation and no user-image upload were performed.

## Results and rerun

`results/{panel_id}.json` contains all 64 raw queries, each background-gate decision,
and the surviving points, in input-crop pixels. `results/summary.json` accounts for
all 178 shared BW panels. `results/runtime.json` records runtime/model provenance.
`weights/provenance.json` records the downloaded official artifact.

Run from the repository root with the existing local Python runtime:

```powershell
python experiments/published_chart_models/lineex/run_keypoints.py --smoke
python experiments/published_chart_models/lineex/run_keypoints.py
python -m unittest discover -s experiments/published_chart_models/lineex -p 'test_*.py'
```

Results are cached only when the input hash matches. `--overwrite` explicitly
recomputes the same model. There is no confidence threshold tuning, training,
image-specific fallback, or production v46 code modification in this experiment.

## Color extension

The identical checkpoint/wrapper also runs on `../inputs_color/manifest.json`,
with output in `results_color/`. The color scope is 314 panels: all 274 panels
eligible in the archived color v46 run, plus 40 no-legend line panels. Those 40
panels are allowed for the published keypoint modules but remain explicitly
unavailable for archived v46 before/after comparison. Previously failed v46 panels
are not dropped. Fourteen BW panels and 92 unsupported panels are excluded and
accounted for in `../inputs_color/coverage.json`.

Native color crops were verified pixel-for-pixel against the unchanged original
source images. The ROI is the actual archived integer-inclusive `plot_area` /
`legend_area`, converted to half-open coordinates, not the unrounded floating
review box. No rotation, resizing, or masking is applied in preparing these crops.
The usual 288x384 warp still occurs inside LineEX as prescribed by its inference.

All 314 color inferences completed. Eight color-input/export tests passed. The
seven color smoke cases match the unmodified official `keypoints()` output in
exact ordered pixel coordinates. See `results_color/upstream_parity.json` and
`../inputs_color/audit.json`; neither report is a ground-truth accuracy benchmark.
