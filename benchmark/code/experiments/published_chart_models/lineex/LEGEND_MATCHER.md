# Official neural legend matcher

`legend_matcher.py` reuses the two official LineEX architecture definition files
and public trained checkpoints. It implements the learned similarity in
`modules/Grouping_legend_mapping/grouping.py`; unlike the repository's top-level
`pipeline.py`, it **does not overwrite the neural score with a color histogram**.
This is the LineEX neural-matching component referred to by AI-ChartParser
Section 3.3.2, not a complete implementation of either chart parser by itself.

## API

```python
from legend_matcher import LegendMatcher, extract_pivot_patches

matcher = LegendMatcher(device="cuda:0", batch_size=16)
extracted = extract_pivot_patches(native_image_rgb, corrected_points_xy)
scores = matcher.score_patches(legend_rgb_crops, extracted["patches"])
# scores: NumPy float32 [number of valid pivots, number of legends]
result = matcher.match(legend_rgb_crops, extracted["patches"])
# result: scores, legend_indices (argmax), winning_scores, status
```

`extract_pivot_patches()` returns `patches`, `valid_indices`, `boxes`, and
`unavailable`. Patches are exactly 40 pixels wide by 20 high, in the supplied
native image coordinates. Bounds are `int(x-20), int(y-10), int(x+20), int(y+10)`,
as in upstream code. Incomplete boundary patches are explicitly skipped, not
padded or wrapped. `valid_indices` maps matcher rows back to corrected point IDs.

Legend patches are the supplied native RGB legend-image crops; the caller must
provide their bounding boxes. This component neither finds legends nor borrows
v46 point labels. Zero legends produces indices `-1` and status `no_legend`,
never a fabricated series. `winning_scores` then contains NaN, which callers must
convert to `null` or omit when writing strict JSON.

`embed_patches()` and `score_embeddings()` are available for caching. Their arrays
are respectively `[N,100]` and `[P,L]`. `.provenance` is JSON-serializable metadata.

## Exact preprocessing and network

Both legend and pivot patches are **RGB**, not the BGR convention used by the
separate LineEX keypoint detector. The transform is the authors' PIL
`Resize(224)` (shorter edge, bilinear), `CenterCrop(224)`, `ToTensor()`, then
ImageNet mean `[0.485,0.456,0.406]` and standard deviation
`[0.229,0.224,0.225]`. A 40x20 patch consequently resizes to 448x224 and is
center-cropped to 224x224. There is no marker-scale fitting or altered crop rule.

The author's function called `resnet101()` is a torchvision ResNet using
**BasicBlock [3,4,23,3]**, not the usual ResNet-101 Bottleneck architecture.
`EmbeddingNet` outputs 100 values. `legend_network(200)` concatenates the legend
embedding FIRST and pivot embedding SECOND, and applies 200→100 Linear, ReLU,
dropout disabled in evaluation, 100→1 Linear, sigmoid. Scores are not cosine
distances and are not calibrated probabilities that a marker is correct.
The highest score assigns each valid pivot to one legend; no threshold is added.

Repeated identical TripletNet branches are evaluated only once per patch, which
is the same shared embedding module in evaluation mode. Pair scores are batched.
Inference uses float32 with TF32 disabled inside a temporary context, restoring
caller backend settings afterward. This removes batch-shape-dependent reduced
precision differences without changing the model, weights, or matching rule.

## Checkpoints and verification

Only the following two public files were downloaded, not the whole Drive folder:

| File | Public Drive ID | Bytes | Epoch | Strict state keys |
| --- | --- | ---: | ---: | ---: |
| `ckpt_30.t7` | `1N49INpPKXWbAk7DCk8dRJX9MUnUa184d` | 165,916,917 | 31 | 422 |
| `mlp_ckpt.t7` | `1FtA-gJoU6Nf1nUTkMX2g_909l1rn96N0` | 246,529 | 50 | 4 |

Downloaded from the authors' linked public folder:
https://drive.google.com/drive/folders/15bKREf2EBORHBZx_xz53eVJzs9e4cZ7Q

Source commit: `b5776d501696087894613fc7accf2180f98f89fe`. SHA256 checksums and
download URLs are in `weights/matching_provenance.json`. Loading uses
`weights_only=True`, rejects unexpected unsafe globals, and requires every state
key to match strictly. No global packages or source/vendor files were modified.

`test_legend_matcher.py` checks shape/edge handling, exact RGB preprocessing,
checkpoint keys, no-legend behavior, and the same official TripletNet and MLP
equations evaluated one pair at a time. `matcher_checks/smoke_parity.json` records
the numerical comparison and complete provenance. These synthetic smoke tests
are implementation verification, not evidence of accuracy on the user's plots.
