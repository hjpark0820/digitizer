"""Run official ChartParser's released network heads locally (not full digitization).

The public single_img_demo is the reference for 512-square BGR preprocessing,
last-stack sigmoid heatmap, 0.3 point cutoff and 0.4/0.45 detector NMS.  Its
tools2 imports and OCR/grouping calls are missing or disabled.  This wrapper
does not replace those missing pieces with our detector or invented grouping.
The released weights contain six anchors per scale, whereas the current source
configuration declares three: reconstruct this dimension from saved anchor_grid,
then load ALL 1939 state-dict tensors strictly.  No weights are dropped.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT.parent / 'vendor' / 'ChartParser'
sys.path[:0] = [str(ROOT / 'python_deps'), str(VENDOR)]
sys.dont_write_bytecode = True
os.environ.setdefault('MPLBACKEND', 'Agg')
import cv2
import numpy as np
from PIL import Image
import torch

COMMIT = '1ce7ee206b4b97fefdce265fbab22de6373988a7'
WEIGHT_SHA256 = '06f8cd58e9a46e1a66a13bb6c982671ade3b903eec6c430adfc45378ebdeba87'
CLASS_NAMES = {0: 'Y_title', 1: 'chart_title', 2: 'x_title', 3: 'unmapped_class_3',
               4: 'x_trick', 5: 'y_trick', 6: 'legend_marker', 7: 'legend_txt'}
WARNINGS = [
    'Official released neural-network heads only; OCR and series grouping disabled/missing upstream.',
    'Keypoints are untyped/unassigned to series. No calibrated axis data or reconstructed series are claimed.',
    'Every native panel crop is stretched to 512x512, preserving upstream single-image BGR preprocessing.',
    'Checkpoint has 6 detector anchors per scale, source has 3: construct head from saved anchor_grid, then strict-load all weights.',
    'Plot ROI filtering is external evaluation/display scope, not an upstream detector operation. Raw predictions retained.',
    'Repository README names a different title while citing AI-ChartParser (CGF 2025); results reproduce this exact public repository/checkpoint.',
]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')

def in_box(x, y, box):
    return bool(box and box[0] <= x < box[2] and box[1] <= y < box[3])

def build_model(weights, device):
    start = time.perf_counter()
    assert digest(weights) == WEIGHT_SHA256, 'Checkpoint hash differs from official downloaded artifact'
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    state = {k.replace('module.model.', 'model.'): v for k, v in checkpoint['state_dict'].items()}
    assert all(isinstance(v, torch.Tensor) for v in state.values())
    module = importlib.import_module('lib.models.ChartParser')
    blocks = copy.deepcopy(module.ChartParser)
    saved = state['model.24.anchor_grid']
    assert tuple(saved.shape) == (3, 1, 6, 1, 1, 2)
    anchors = saved.reshape(3, -1).tolist()
    assert blocks[25][1].__name__ == 'Detect'
    original_anchors = blocks[25][2][1]
    blocks[25][2][1] = anchors
    model = module.ChartParserModel(blocks)
    incompatible = model.load_state_dict(state, strict=True)
    assert not incompatible.missing_keys and not incompatible.unexpected_keys
    params = sum(p.numel() for p in model.parameters())
    model.eval().to(device)
    del checkpoint, state
    info = {'commit': COMMIT, 'weights_path': str(weights), 'weights_sha256': WEIGHT_SHA256,
            'state_dict_keys_strictly_loaded': 1939, 'parameters': params,
            'original_config_anchors': original_anchors, 'checkpoint_config_anchors': anchors,
            'torch': torch.__version__, 'device': str(device),
            'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'model_load_seconds': time.perf_counter() - start,
            'warnings': WARNINGS, 'license': 'No LICENSE file present in pinned public repository; do not redistribute as licensed code.'}
    return model, info

def infer(panel, model, device, output):
    from tools.decode_heatmap import decode_hm
    from lib.core.general import non_max_suppression
    start = time.perf_counter()
    inp = Path(panel['image_path'])
    assert digest(inp) == panel['input_sha256']
    bgr = cv2.imdecode(np.fromfile(inp, np.uint8), cv2.IMREAD_COLOR)
    assert bgr is not None
    height, width = bgr.shape[:2]
    assert [width, height] == panel['image_size']
    resized = cv2.resize(bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    # Same as torchvision ToTensor + Normalize on upstream cv2 BGR uint8.
    tensor = torch.from_numpy(resized.transpose(2, 0, 1).copy()).float().div_(255)
    tensor = (tensor - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
    tensor = tensor.unsqueeze(0).to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    net_start = time.perf_counter()
    with torch.inference_mode():
        det_out, stacks, segmentation = model(tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    network_seconds = time.perf_counter() - net_start
    # CPU decoder avoids the upstream device-string ambiguity but keeps all maths.
    heatmap = stacks[-1]['hm'].sigmoid().cpu()
    offsets = stacks[-1]['reg'].cpu()
    decoded = decode_hm(heatmap, offsets, .3, 'cpu')[0].numpy()
    assert heatmap.shape == (1, 1, 128, 128)
    raw_points = []
    for hx, hy, score, klass in decoded:
        # Upstream single_img_demo casts coordinates to int in 512-space.
        nx, ny = int(float(hx) * 4), int(float(hy) * 4)
        x, y = nx / 512 * width, ny / 512 * height
        keep = in_box(x, y, panel['plot_box']) and not in_box(x, y, panel.get('legend_box'))
        raw_points.append({'x': x, 'y': y, 'score': float(score), 'class_id': int(klass),
                           'model_x': nx, 'model_y': ny,
                           'subpixel_x': float(hx) / 128 * width,
                           'subpixel_y': float(hy) / 128 * height,
                           'inside_plot_excluding_legend': keep, 'series_id': None})
    boxes_tensor = non_max_suppression(det_out[0].cpu(), conf_thres=.4, iou_thres=.45)[0]
    boxes = []
    for x1, y1, x2, y2, conf, klass in boxes_tensor.numpy():
        # Upstream boxes rounded in 512-space before original image rescale.
        box = [float(round(float(x1))) / 512 * width, float(round(float(y1))) / 512 * height,
               float(round(float(x2))) / 512 * width, float(round(float(y2))) / 512 * height]
        boxes.append({'xyxy': box, 'score': float(conf), 'class_id': int(klass),
                      'class_name': CLASS_NAMES[int(klass)]})
    mask512 = segmentation.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
    assert mask512.shape == (512, 512)
    paneldir = output / panel['panel_id']
    paneldir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask512 * 255).save(paneldir / 'curve_mask_512.png')
    native_mask = cv2.resize(mask512, (width, height), interpolation=cv2.INTER_NEAREST)
    Image.fromarray(native_mask * 255).save(paneldir / 'curve_mask.png')
    np.savez_compressed(paneldir / 'raw_heads.npz', heatmap=heatmap.numpy(), offsets=offsets.numpy(),
                        segmentation=segmentation.cpu().numpy(), decoded_keypoints=decoded,
                        boxes=boxes_tensor.numpy())
    points = [p for p in raw_points if p['inside_plot_excluding_legend']]
    result = {'panel_id': panel['panel_id'], 'status': 'completed', 'points': points,
              'raw_points': raw_points, 'coordinate_system': 'input_crop_pixels',
              'curve_mask_path': str(paneldir / 'curve_mask.png'),
              'curve_mask_model_path': str(paneldir / 'curve_mask_512.png'),
              'raw_heads_path': str(paneldir / 'raw_heads.npz'), 'boxes': boxes,
              'model_input_size': [512, 512], 'heatmap_size': [128, 128],
              'input_image_size': [width, height], 'input_sha256': panel['input_sha256'],
              'plot_box': panel['plot_box'], 'legend_box': panel.get('legend_box'),
              'timing': {'network_seconds': network_seconds, 'total_seconds': time.perf_counter() - start},
              'warnings': WARNINGS, 'physical_data': None, 'reconstructed_series': None,
              'keypoint_threshold': .3, 'detector_confidence_threshold': .4, 'detector_nms_iou_threshold': .45}
    save_json(paneldir / 'result.json', result)
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT.parent / 'inputs' / 'manifest.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    parser.add_argument('--weights', type=Path, default=ROOT / 'weights' / 'epoch-19.pth')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.manual_seed(0)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    panels = manifest['panels']
    if args.smoke:
        panels = [p for p in panels if p['panel_id'] in manifest['smoke_ids']]
    if args.limit:
        panels = panels[:args.limit]
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    model, provenance = build_model(args.weights, device)
    save_json(args.output / 'provenance.json', provenance)
    print(json.dumps(provenance), flush=True)
    results = []
    for panel in panels:
        resultpath = args.output / panel['panel_id'] / 'result.json'
        if resultpath.exists():
            result = json.loads(resultpath.read_text(encoding='utf-8'))
            assert result.get('input_sha256') == panel['input_sha256']
            print(f"{panel['panel_id']} retained existing {result['status']}", flush=True)
        else:
            try:
                result = infer(panel, model, device, args.output)
            except Exception:
                result = {'panel_id': panel['panel_id'], 'status': 'failed', 'points': [],
                          'input_sha256': panel['input_sha256'], 'error': traceback.format_exc(),
                          'coordinate_system': 'input_crop_pixels', 'warnings': WARNINGS}
                save_json(resultpath, result)
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            print(f"{panel['panel_id']}: {result['status']}, {len(result['points'])} scoped points, "
                  f"{len(result.get('raw_points', []))} raw points", flush=True)
        results.append({'panel_id': result['panel_id'], 'status': result['status'],
                        'point_count': len(result['points']), 'result_path': str(resultpath),
                        'timing': result.get('timing')})
        save_json(args.output / ('smoke_summary.json' if args.smoke else 'summary.json'),
                  {'completed_this_run': len(results), 'expected_this_run': len(panels), 'panels': results})

if __name__ == '__main__':
    main()
