# Standalone v46 runtime

The supported web/CLI runtime no longer reads, imports, compiles or requires
`run_A4_auto_v45.py`. Historical v45 files in the research workspace are not
included in the standalone distribution and can be absent at runtime.

OCR calls use `ocr_bridge_v46.py`. The optional `--ai-ocr` server/packaged-launcher
flag enables a local snippet/answer handoff when Tesseract is unavailable. See
[AI_OCR.md](AI_OCR.md) for the operating agent's required queue workflow. The
agent must view the snippets and submit answers; no vision API is called by Python.

## Execution flow

1. `unified_server._pick_color_pipeline()` selects `run_A4_auto_v46.py` explicitly.
   Missing v46 is an error, not an invitation to choose another numbered runner.
   A `PLOT_PIPELINE` override may point to another copy of that v46 entry point;
   a stale override naming an older runner is rejected.
2. `run_A4_color_v46.main()` handles optional-legend routing and calls
   `color_pipeline_v46.run()` directly for the legend-driven colour path.
3. `color_pipeline_v46` prepares the image/palette, calls `LegendRuntime.prepare`
   and `MarkerRuntime.run`, then calibrates and exports results. The calibration
   guard and identity-preserving label/empty-series hooks are explicit calls.
4. `chart_analysis_v46` contains shared pixel, colour, legend, calibration and
   rendering functions. Importing the library has no CLI/image-processing effects.
5. `legend_palette_v46` exposes palette filtering, achromatic recovery, palette
   locking and swatch selection as named functions. The colour workflow and
   `legend_optional_v46.native_legend.extract()` share those functions.
6. `AnalysisSession` gives each image a separate module namespace. The shared
   numerical helpers still use image/palette globals, so stage functions are
   bound to that private namespace using `FunctionType`. Normal Python module
   loading replaces runtime source extraction, AST edits and source-line slicing.

`LegendRuntime.colour_masks()` passes prepared samples through the explicit
`samples=` parameter. It never inspects or rewrites a function's source.
`bw_pipeline_v46` imports shared helpers directly as well.

## Intentional compatibility changes

- Missing usable hybrid marker evidence raises a v46 diagnostic. There is no
  fallback to the older colour/walk detector. For marker-free charts, choose
  Auto or line-only and supply the correct legend/plot rectangles.
- The old `--no-color-pipeline` switch and `compile_v45_bridge()` API are removed.
- Optional output directories now use the `_v46_out` suffix.
- Existing image/JSON history, manual edits and saved correction evidence keep
  their formats. The web editor implementation is unchanged by this refactor.
- Historical experiment scripts which intentionally inspect v45 internals are
  outside the supported standalone runtime.

## Validation

A relocated package containing no v45 Python files imports all 141 modules and
starts the server from an unrelated working directory in a clean CPU environment.
Five generated cases cover colour/B&W, with/without a legend, and marker-free
colour lines. Initial detection and one saved-only correction pass for each case.
Before/after comparisons match points, labels, colours, geometry and calibration.
History import/export preserves 55,000 points in a 1,472,804-byte multipart JSON
file, and Excel contains all 55,000 rows. These are regression/smoke checks, not
an accuracy claim for arbitrary real charts.

Focused migration regressions: 83 passed, including ten real legend fixtures.
The broader optional-legend suite passes 111/113; the two failures still expect
`bw_step5_v46` instead of the existing `bw_series_correction_v46` engine name.
Both failures were reproduced on the untouched pre-migration runtime.
