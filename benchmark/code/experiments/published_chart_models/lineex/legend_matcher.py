"""Official LineEX learned legend/point-patch similarity, without color override.

This is the neural matcher in the authors' grouping.py: TripletNet(resnet101())
embeddings followed by legend_network(200), with legend embedding concatenated
FIRST. The model named resnet101 uses BasicBlock [3,4,23,3] in that repository;
substituting torchvision.models.resnet101 would change the trained architecture.

Inputs must be RGB patches. This is intentionally different from the keypoint
module, whose published inference consumes BGR/255. No images are uploaded.
"""
from __future__ import annotations

import hashlib
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT.parent / "vendor" / "LineEX"
AUTHOR_SOURCE = VENDOR / "modules/Grouping_legend_mapping/legend_models"
SOURCE_COMMIT = "b5776d501696087894613fc7accf2180f98f89fe"
WEIGHTS = {
    "embedding": {"name": "ckpt_30.t7", "size": 165916917,
                  "sha256": "917d80699ad98f304cf22eab478d94f37981fbc5ab3b9af8859b90ff5895d862",
                  "file_id": "1N49INpPKXWbAk7DCk8dRJX9MUnUa184d"},
    "matching_mlp": {"name": "mlp_ckpt.t7", "size": 246529,
                     "sha256": "b596d352cf4d8f4d44cb7ccdfef3e9439321c353d2bf17e092a9dcdde927e458",
                     "file_id": "1FtA-gJoU6Nf1nUTkMX2g_909l1rn96N0"},
}


@contextmanager
def full_precision():
    """Avoid TF32 batch-shape numerical differences; restore caller settings."""
    matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with torch.backends.cudnn.flags(allow_tf32=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul_tf32


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def author_module(name, path):
    """Import just the two audited definition files, never upstream run scripts."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def as_rgb_pil(patch):
    if isinstance(patch, Image.Image):
        image = patch.convert("RGB")
    else:
        array = np.asarray(patch)
        if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
            raise ValueError("NumPy patches must be uint8 RGB arrays shaped HxWx3, not BGR or float tensors")
        image = Image.fromarray(array, "RGB")
    if image.width < 1 or image.height < 1:
        raise ValueError("Empty patch")
    return image


def extract_pivot_patches(image_rgb, points_xy):
    """Return exact 40x20 native patches and explicit unavailable edge cases.

    Bounds use int(x-20),int(y-10),int(x+20),int(y+10), matching the authors'
    slicing rule (truncation toward zero, no rounding or image scaling). Outside
    or partial patches are rejected, never negative-index wrapped or padded.
    Valid patch positions correspond to ``valid_indices`` in the original list.
    ``boxes`` is parallel to patches; each entry is half-open native xyxy.
    """
    image = np.asarray(as_rgb_pil(image_rgb))
    height, width = image.shape[:2]
    patches, valid_indices, boxes, unavailable = [], [], [], []
    for index, point in enumerate(points_xy):
        try:
            x, y = float(point[0]), float(point[1])
            if not np.isfinite([x, y]).all():
                raise ValueError("nonfinite coordinate")
            box = [int(x-20), int(y-10), int(x+20), int(y+10)]
            a, b, c, d = box
            if not (0 <= a < c <= width and 0 <= b < d <= height and c-a == 40 and d-b == 20):
                raise ValueError("incomplete 40x20 patch at image boundary")
            patches.append(image[b:d, a:c].copy())
            valid_indices.append(index)
            boxes.append(box)
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            unavailable.append({"index": index, "reason": str(exc)})
    return {"patches": patches, "valid_indices": valid_indices, "boxes": boxes,
            "unavailable": unavailable, "patch_size": [40, 20],
            "coordinate_convention": "native half-open integer xyxy, exact fixed size; no rescaling/padding"}


class LegendMatcher:
    """Reusable official learned matcher; arrays returned on CPU.

    score_patches(legends_rgb, pivots_rgb) -> float32 array [P,L].
    match(...) -> scores, legend_indices [P] argmax, winning_scores [P].
    Zero legends returns shape [P,0], indices=-1, scores=NaN; the API does not
    invent a single series when no legend exists. Zero pivots is supported.
    The sigmoid score is a learned similarity, NOT calibrated correctness.
    """

    def __init__(self, device=None, batch_size=16, weight_dir=None):
        self.device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested but is unavailable")
        if isinstance(batch_size, bool) or int(batch_size) != batch_size or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.batch_size = int(batch_size)
        folder = Path(weight_dir) if weight_dir is not None else ROOT / "weights"
        network = author_module("lineex_official_embedding_net", AUTHOR_SOURCE / "net.py")
        mlp = author_module("lineex_official_legend_mlp", AUTHOR_SOURCE / "MLP.py")
        self.emb_model = network.TripletNet(network.resnet101(pretrained=False))
        self.mlp = mlp.legend_network(200)
        loaded = {}
        for name, model in [("embedding", self.emb_model), ("matching_mlp", self.mlp)]:
            info = WEIGHTS[name]
            path = folder / info["name"]
            if not path.is_file() or path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
                raise ValueError(f"Official checkpoint missing or hash mismatch: {path}")
            unsafe = torch.serialization.get_unsafe_globals_in_checkpoint(path)
            if unsafe:
                raise ValueError(f"Unexpected unsafe checkpoint globals: {unsafe}")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            result = model.load_state_dict(checkpoint["state_dict"], strict=True)
            loaded[name] = {**info, "path": str(path.resolve()), "epoch": checkpoint.get("epoch"),
                            "state_keys": len(checkpoint["state_dict"]), "strict_load_result": str(result),
                            "unsafe_globals": unsafe,
                            "url": f"https://drive.google.com/file/d/{info['file_id']}/view"}
            del checkpoint
            model.eval().to(self.device)
        self.transform = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.provenance = {
            "method": "Official LineEX learned neural legend matching (no color-histogram override)",
            "source_repository": "https://github.com/Shiva-sankaran/LineEX", "source_commit": SOURCE_COMMIT,
            "source_definition_sha256": {name: sha(AUTHOR_SOURCE / name) for name in ["net.py", "MLP.py"]},
            "checkpoints": loaded, "device": str(self.device), "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__, "batch_size": self.batch_size,
            "embedding_architecture": "Authors' resnet101: torchvision ResNet BasicBlock [3,4,23,3], 512->100 embedding",
            "embedding_parameters": sum(p.numel() for p in self.emb_model.parameters()),
            "matching_parameters": sum(p.numel() for p in self.mlp.parameters()),
            "matching_architecture": "concat(legend100,pivot100) -> Linear200x100 -> ReLU -> Dropout0.2 disabled in eval -> Linear100x1 -> Sigmoid",
            "preprocessing": "RGB PIL; Resize224 shorter edge with bilinear; CenterCrop224; ToTensor/255; ImageNet mean [0.485,0.456,0.406], std [0.229,0.224,0.225]",
            "native_pivot_patch_size": [40, 20], "similarity_shape": "P pivots x L legends",
            "histogram_override": False, "weights_strict": True,
            "numeric_precision": "float32; TF32 disabled within inference contexts, caller backend settings restored",
            "implementation_note": "One embedding per image replaces three identical eval-mode TripletNet branch evaluations; no change to learned weights or scoring order.",
        }

    @torch.inference_mode()
    def embed_patches(self, patches: Iterable):
        patches = list(patches)
        if not patches:
            return np.empty((0, 100), dtype=np.float32)
        arrays = []
        with full_precision():
            for start in range(0, len(patches), self.batch_size):
                tensor = torch.stack([self.transform(as_rgb_pil(p)) for p in patches[start:start+self.batch_size]]).to(self.device)
                # TripletNet.forward(a,a,a)[0] is exactly this same shared module in eval.
                embedding = self.emb_model.embeddingnet(tensor)
                arrays.append(embedding.cpu().numpy())
        output = np.concatenate(arrays, axis=0)
        if not np.isfinite(output).all():
            raise ValueError("Nonfinite official embedding")
        return output

    @torch.inference_mode()
    def score_embeddings(self, legend_embeddings, pivot_embeddings):
        legends = np.asarray(legend_embeddings, dtype=np.float32)
        pivots = np.asarray(pivot_embeddings, dtype=np.float32)
        if legends.ndim != 2 or pivots.ndim != 2 or legends.shape[1] != 100 or pivots.shape[1] != 100:
            raise ValueError("Both embedding arrays must have shape N x 100")
        if not np.isfinite(legends).all() or not np.isfinite(pivots).all():
            raise ValueError("Nonfinite input embedding")
        n_pivots, n_legends = len(pivots), len(legends)
        scores = np.empty((n_pivots, n_legends), dtype=np.float32)
        if not n_pivots or not n_legends:
            return scores
        legend_tensor = torch.from_numpy(legends).to(self.device)
        with full_precision():
            for start in range(0, n_pivots, self.batch_size):
                pivot_tensor = torch.from_numpy(pivots[start:start+self.batch_size]).to(self.device)
                count = len(pivot_tensor)
                lhs = legend_tensor[None].expand(count, -1, -1).reshape(-1, 100)
                rhs = pivot_tensor[:, None].expand(-1, n_legends, -1).reshape(-1, 100)
                scores[start:start+count] = self.mlp(lhs, rhs).reshape(count, n_legends).cpu().numpy()
        if not np.isfinite(scores).all():
            raise ValueError("Nonfinite learned similarity")
        return scores

    def score_patches(self, legend_images_rgb, pivot_patches_rgb):
        legends = self.embed_patches(legend_images_rgb)
        pivots = self.embed_patches(pivot_patches_rgb)
        return self.score_embeddings(legends, pivots)

    def match(self, legend_images_rgb, pivot_patches_rgb):
        scores = self.score_patches(legend_images_rgb, pivot_patches_rgb)
        n_pivots, n_legends = scores.shape
        if n_legends:
            indices = scores.argmax(axis=1).astype(np.int64)
            winning = scores[np.arange(n_pivots), indices]
        else:
            indices = np.full(n_pivots, -1, dtype=np.int64)
            winning = np.full(n_pivots, np.nan, dtype=np.float32)
        return {"scores": scores, "legend_indices": indices, "winning_scores": winning,
                "status": "completed" if n_legends else "no_legend", "threshold_applied": False}
