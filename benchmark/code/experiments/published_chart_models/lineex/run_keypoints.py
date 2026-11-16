"""Local LineEX L+D keypoint-module reproduction (not the full OCR pipeline).

Uses unmodified upstream architecture and background gate with official weights.
The factory bridge only avoids newer timm's extra unsupported factory arguments.
No v46 marker estimates or reference annotations are supplied to the network.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
VENDOR = EXPERIMENT / "vendor" / "LineEX"
WEIGHT = ROOT / "weights" / "ckpt_L+D.t7"
WEIGHT_SHA256 = "fb7f1ed279e30e4916f2c02e14ebf2457a3bf4cdb40452e05c8d67702c3023a6"
INPUT_SIZE = [288, 384]  # cv2 width,height, exactly as upstream keypoints().


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_model(device: str):
    if digest(WEIGHT) != WEIGHT_SHA256:
        raise RuntimeError("LineEX checkpoint hash mismatch")
    sys.path.insert(0, str(VENDOR / "modules" / "KP_detection"))
    from models.my_model import Model
    from models import transformer_vit, xcit
    from utils import checkifbackground

    def official_factory(name, **kwargs):
        if name != "xcit_small_12_p16":
            raise ValueError(name)
        # timm 0.5.4 removed None values. timm 1.x adds pretrained_cfg args;
        # direct invocation of the SAME registered upstream function avoids them.
        return xcit.xcit_small_12_p16(**{k: v for k, v in kwargs.items() if v is not None})

    transformer_vit.create_model = official_factory
    args = SimpleNamespace(
        patch_size=16, input_size=INPUT_SIZE, vit_dropout=0., vit_weights=None,
        hidden_dim=384, nheads=8, dim_feedforward=1536, dropout=0., activation="gelu",
        pre_norm=False, dec_layers=6, vit_dim=384, position_embedding="enc_xcit", num_queries=64,
    )
    model = Model(args)
    checkpoint = torch.load(WEIGHT, map_location="cpu", weights_only=True)
    match = model.load_state_dict(checkpoint["state_dict"], strict=True)
    metadata = {
        "checkpoint_sha256": WEIGHT_SHA256,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_state_keys": len(checkpoint["state_dict"]),
        "strict_load_result": str(match),
        "parameter_count": sum(p.numel() for p in model.parameters()),
    }
    del checkpoint
    return model.eval().to(device), checkifbackground, metadata


def inside(x, y, box):
    return box is not None and box[0] <= x < box[2] and box[1] <= y < box[3]


def read_image(path: Path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode image {path}")
    return image


def infer(model, background_gate, panel, device):
    started = time.perf_counter()
    path = Path(panel["image_path"])
    image = read_image(path)
    h, w = image.shape[:2]
    image_hash = digest(path)
    resized = cv2.resize(image, tuple(INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    # Deliberately BGR/255: RGB normalization would NOT reproduce upstream.
    tensor = torch.from_numpy(resized.astype(np.float32) / 255).permute(2, 0, 1)[None].to(device)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    network_start = time.perf_counter()
    with torch.inference_mode():
        normalized = model(tensor, return_attn=False)["pred_boxes"][0].cpu().numpy()
    network_seconds = time.perf_counter() - network_start
    assert normalized.shape == (64, 2) and np.isfinite(normalized).all()
    raw, points, errors = [], [], []
    for query_id, (nx, ny) in enumerate(normalized):
        # Upstream truncates normalized coordinates to uint32. Python ints
        # preserve NumPy 1.x signed slicing arithmetic in the background gate.
        x = int(np.uint32(nx * w))
        y = int(np.uint32(ny * h))
        record = {
            "query_id": query_id, "x": x, "y": y, "score": None,
            "normalized_x": float(nx), "normalized_y": float(ny),
            "in_plot": inside(x, y, panel.get("plot_box")),
            "in_legend": inside(x, y, panel.get("legend_box")),
        }
        try:
            background = bool(background_gate((x, y), image))
            record.update(background_rejected=background, gate_status="completed")
            if not background:
                points.append(record.copy())
        except Exception as exc:
            record.update(background_rejected=None, gate_status="failed", gate_error=str(exc))
            errors.append(f"query {query_id}: {exc}")
        raw.append(record)
    return {
        "panel_id": panel["panel_id"], "status": "completed" if not errors else "partial_gate_failure",
        "method": "LineEX L+D official keypoint module",
        "pipeline_scope": "keypoint_module_only_no_grouping_no_ocr_no_axis_scaling",
        "coordinate_system": "input_crop_pixels", "image_path": str(path),
        "image_sha256": image_hash, "source_sha256": panel.get("source_sha256"),
        "image_size": [w, h], "model_input_size": INPUT_SIZE,
        "input_preprocessing": "OpenCV BGR, bilinear warp to width288 height384, float32/255, no mean/std normalization",
        "points": points, "raw_queries": raw, "raw_query_count": len(raw),
        "background_rejected_count": sum(r.get("background_rejected") is True for r in raw),
        "plot_point_count": sum(r["in_plot"] and not r["in_legend"] for r in points),
        "timing": {"network_seconds": network_seconds, "total_seconds": time.perf_counter() - started},
        "warnings": ["Keypoints have no confidence head and no series/marker-shape labels.",
                     "The official model emits at most 64 queries per input crop.",
                     "The model internally warps the native crop to 288x384; outputs are restored to native crop coordinates."] + errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EXPERIMENT / "inputs" / "manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = False
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    panels = manifest["panels"]
    if args.smoke:
        ids = set(manifest["smoke_ids"])
        panels = [p for p in panels if p["panel_id"] in ids]
    args.output.mkdir(parents=True, exist_ok=True)
    setup_start = time.perf_counter()
    model, gate, metadata = load_model(args.device)
    metadata.update(
        device=args.device, torch_version=torch.__version__, opencv_version=cv2.__version__,
        source_repository="https://github.com/Shiva-sankaran/LineEX",
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=VENDOR, text=True).strip(),
        source_changed_files=subprocess.check_output(["git", "status", "--porcelain"], cwd=VENDOR, text=True).splitlines(),
        setup_seconds=time.perf_counter()-setup_start,
        compatibility_changes=["Directly call the official registered XCiT constructor instead of timm1.x factory.",
                               "Cast query pixel scalars to Python int for NumPy1.x-compatible signed slicing.",
                               "Use torch.load(weights_only=True), strict state_dict loading, inference_mode."])
    (args.output / "runtime.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    summary = []
    for number, panel in enumerate(panels, 1):
        result_path = args.output / f"{panel['panel_id']}.json"
        if result_path.exists() and not args.overwrite:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("image_sha256") != digest(Path(panel["image_path"])):
                raise RuntimeError(f"Cached input changed: {panel['panel_id']}")
        else:
            try:
                result = infer(model, gate, panel, args.device)
            except Exception as exc:
                result = {"panel_id": panel["panel_id"], "status": "failed", "reason": str(exc), "points": []}
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        summary.append({"panel_id": panel["panel_id"], "status": result["status"],
                        "points": len(result["points"]), "plot_points": result.get("plot_point_count"),
                        "timing": result.get("timing"), "result_path": str(result_path)})
        print(f"{number}/{len(panels)} {panel['panel_id']} {result['status']} points={len(result['points'])} plot={result.get('plot_point_count')}", flush=True)
    (args.output / ("smoke_summary.json" if args.smoke else "summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
