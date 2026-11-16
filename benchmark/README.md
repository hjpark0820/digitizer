# Benchmark: v46 against AI-ChartParser and LineEX on 100 synthetic PK / ER charts (test100v2)

This folder holds everything needed to check the benchmark reported in the paper: the 100 test charts and their
ground truth, every method's predictions, the scoring code, and the code that makes the figures and tables.

## 1. Result (pooled F1, %, with 95 % chart-bootstrap interval)

| method | PK B&W | PK colour | ER B&W | ER colour | all 100 |
|---|---:|---:|---:|---:|---:|
| v46, after Step 5 | 69.4 | 86.4 | 75.3 | 94.5 | **81.7** (77.8–85.4) |
| v46, before Step 5 | 68.5 | 86.0 | 74.2 | 93.7 | 80.9 (77.0–84.6) |
| AI-ChartParser, raw keypoints | 28.8 | 52.6 | 34.5 | 65.5 | 45.4 (40.9–49.9) |
| AI-ChartParser, corrected keypoints | 26.5 | 52.1 | 30.0 | 61.1 | 42.8 (38.2–47.4) |
| LineEX, paper settings | 23.0 | 37.9 | 27.3 | 39.7 | 32.0 (29.4–34.3) |
| LineEX, released-code settings | 21.5 | 37.1 | 25.0 | 39.2 | 30.6 (28.3–33.0) |
| charts / ground-truth points | 25 / 891 | 25 / 1,057 | 25 / 568 | 25 / 576 | 100 / 3,092 |

PK = concentration-time charts, ER = exposure-response charts, B&W = black-and-white. Precision, recall, TP / FP / FN,
paired differences, per-chart scores, the Step 5 effect and the legend-input sensitivity analyses are in `tables/`;
figures in `figures/` (PNG 300 dpi and PDF, draft captions in `figures/figure_captions.md`).

## 2. Test set

100 synthetic charts, 25 per group (PK B&W, PK colour, ER B&W, ER colour), 3,092 ground-truth marker centres
(`data/benchmark/targets_test100v2_20261025.json`). Each chart is a re-drawing of a published figure: the canvas,
tick labels, legend texts and axis titles follow the source figure, and the data are digitised source points with a
smooth random perturbation; they are not clinical measurements. Charts were drawn at random (seed 20261018) from the
synthetic collection, one variant per source figure, excluding every source figure of the final development set
(`code/test_set/build_test100.py`, `data/benchmark/selection_test100_v1.json`). In version 2, 37 images that had been
used during method development were replaced by another rendering of the same source figure (1 by another source), so
no test image was used during development. The test set is image-disjoint, not source-disjoint, from the development
data: every source figure also appears, as other renderings, in synthetic sets used during development (65 of the 100
in the main development set). `data/benchmark/test100v2_charts.json` lists every chart and whether it was replaced.

## 3. What every method received

- **All methods**: the same image and the same two rectangles a user draws in the v46 GUI — the plot area and the
  legend region (`plot_bbox`, `legend_bbox` in `data/benchmark/inputs_test100v2_20261025.json`).
- **v46** (B&W and colour): only these two rectangles, i.e. the options the web GUI sends; it finds the individual
  legend keys itself. Step 5 (rendering-based correction) runs through the GUI's own route, at most 5 iterations.
  (`code/experiments/published_chart_models/benchmark_v2/run_gui_legend.py` builds exactly this command.)
- **AI-ChartParser** (`ywking/ChartParser` @ `1ce7ee206b4b97fefdce265fbab22de6373988a7`, checkpoint `epoch-19.pth`) and
  **LineEX** (`Shiva-sankaran/LineEX` @ `b5776d501696087894613fc7accf2180f98f89fe`): the whole image and one image crop
  per legend key — the crops v46 exported (unmodified source pixels). They do not localise legend keys themselves; the
  two rectangles only drop detected points outside the plot or inside the legend. Series assignment for both uses the
  released LineEX legend matcher. Where v46 found no legend key at all (1 chart), the crops were cut from the annotated
  key boxes instead. "Paper settings" of LineEX = the two values its paper prints (background gate 0.98, legend crops
  resized to 40 x 20); "released-code settings" = the code as published.
- Shared legend list: 361 entries. Keys with no entry: B&W 9 on 7 charts, colour 5 on 5 charts, so 111 of 3,092
  ground-truth points cannot be matched by any method.
- No method was trained or tuned on these charts.

## 4. Scoring

`code/experiments/published_chart_models/benchmark_v2/evaluate_entry.py`. A predicted point is a true positive when it
lies within the OKS tolerance radius of a ground-truth point (0.0294 x image diagonal, the OKS > 0.5 criterion of
LineEX) and is assigned to the legend entry of that point's series; matching is one-to-one. On the 4 single-series
charts only position counts. Pooled F1 = 2TP / (2TP + FP + FN) over all points of a group; 95 % intervals from 2,000
chart resamples; paired differences resample charts once and score both methods on the same resample (seed 0).

Notes: v46 failed on 2 B&W charts — t1_pk_bw_15 (no legend key found; scored 0) and t1_er_bw_01 (Step 5 failed; scored
0 after Step 5). Colour outputs come from earlier runs of the same code snapshot; B&W v46 ran on CPU, and a CUDA run
gave identical predictions on all 50 B&W charts (`data/checks/backend_checks.json`).

## 5. Legend-input sensitivity (`tables/table4_legend_input_sensitivity.*`, `figures/fig6_*`)

- v46 given the annotated legend-key boxes instead of finding them (B&W): pooled F1 after Step 5 69.9 % vs 71.8 %
  (paired difference -1.8 points, 95 % CI -5.4 to +1.4). The annotated boxes are smaller than the drawn keys.
- AI-ChartParser and LineEX given crops cut at the annotated key boxes instead of v46's crops: all-100 F1 43.4 / 41.2 /
  32.6 / 30.8 (vs 45.4 / 42.8 / 32.0 / 30.6). They recover 29–33 of the 111 points on keys v46 missed but lose more on
  the other keys.

## 6. Reproducing

Python 3.12, numpy, scipy, matplotlib, Pillow, OpenCV (torch / torchvision for the published methods).

Re-score from the stored predictions (no models needed): copy `data/results/v46` and `data/results/external` to an
empty folder OUT, then from `code/experiments/published_chart_models/benchmark_v2/`:
```
python evaluate_entry.py --inputs ../../../../data/benchmark/inputs_test100v2_20261025.json \
    --targets ../../../../data/benchmark/targets_test100v2_20261025.json \
    --results OUT --ground-truth ../../../../data/benchmark/empty_ground_truth.json
```
Figures and tables: `python code/figures/make_figures.py` (reads only this folder).

A full re-run needs the v46 code snapshot `0fd678a840bc` (file hashes: `code/v46_snapshot_0fd678a840bc_fingerprint.json`;
the `src/` of this repository is a later standalone distribution: 204 of the snapshot's code files are identical,
7 changed, 80 are not part of the distribution), the two upstream repositories with their weights (download scripts in
`code/experiments/published_chart_models/{chartparser,lineex}/`; AI-ChartParser's repository has no licence file, so
its code is not included here) and the exact drivers in `code/run_drivers/`.

Local paths in the stored records were replaced by placeholders (`<PROJECT>`, `<BENCHMARK_V2>`, `<SYNTHETIC_DATASET>`,
`<HOME>`, `<PYTHON>`, `<TEMP>`); image paths point to `data/images/`. The scorer reads none of these paths.
`MANIFEST.sha256` lists every file.
