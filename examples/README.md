# Synthetic example plots

All 40 input PNGs are together in this folder: 10 PK B&W, 10 PK colour, 10 ER B&W and 10 ER colour. Filename prefixes identify the categories. PK means pharmacokinetic; ER means exposure-response.

These are deliberately selected easy/high-performing examples, not a representative benchmark. Sixty-two synthetic plots were screened with the current public runtime. The selected 40 are distinct, single-panel examples with final F1 at least 0.95. Following review, the single-series Cusatuzumab2 example was replaced with the two-series Motavizumab p02 example. They are redrawn synthetic charts with perturbed/simulated data, not original clinical measurements.

Start the app from the repository root with `python run.py`, choose an image under **New detection**, leave **Automatic** selected and select the plot and legend regions. `examples_manifest.json` records the exact regions and numeric axis calibration used in verification. Verification supplied these values; automatic OCR calibration was not assessed. Run detection, apply automatic correction, and inspect the result. The test used automatic correction in **Auto** mode, with a 20-iteration cap, rather than the initial **Manual** setting of 5 iterations.

The manifest contains TP (correct), FP (extra/wrong-series), FN (missed), and F1 values before and after automatic correction, together with timing values and the runtime fingerprint. A match requires the correct legend series within 5 source pixels. Reference points were used only after inference, never as detector inputs. Timing includes process startup and up to four concurrent CPU jobs, so it is not an isolated speed benchmark.

The table below summarizes results after automatic correction.

| Category | Images | TP after automatic correction | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PK B&W | 10 | 223 | 1 | 11 | 97.38% |
| PK colour | 10 | 200 | 0 | 2 | 99.50% |
| ER B&W | 10 | 147 | 0 | 1 | 99.66% |
| ER colour | 10 | 195 | 0 | 0 | 100.00% |

The PNGs are unchanged copies of rendered synthetic inputs. Source images were not moved or modified. There are no category subfolders. Full digitization results and screening diagnostics remain outside the public package; `examples_manifest.json` is descriptive metadata, not an editable saved session.
