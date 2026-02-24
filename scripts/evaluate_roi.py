#!/usr/bin/env python3
"""
ROI localization evaluator using the full pipeline_v2 detection stack.

Loads a video (or all videos from roi_manifest.json), buffers frames up to the
target frame, runs the pipeline detection/aggregation/localization chain,
and compares the predicted bird chest ROI against the expert-labeled ground truth.

Usage
-----
Single video:
    python scripts/evaluate_roi.py --config configs/default.yaml \
        --video "/path/to/video.MOV"

All videos in manifest:
    python scripts/evaluate_roi.py --config configs/default.yaml \
        --all

The detector combination used is determined by the config file (keys
`detection` and `segmentation`).  Results are tagged with the config name
and saved under rois/eval_v2/<config_stem>/ so that different detector
combinations can be compared afterwards.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

# Allow running from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#from src.pipeline_v2 import BreathingAnalyzer
from src.pipeline import BreathingAnalyzer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou(box1, box2):
    """IoU of two (x, y, w, h) boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi = max(x1, x2)
    yi = max(y1, y2)
    xr = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    if xr <= xi or yb <= yi:
        return 0.0
    inter = (xr - xi) * (yb - yi)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def center_distance(box1, box2):
    """Euclidean distance between box centres (pixels)."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return float(np.hypot(x1 + w1 / 2 - (x2 + w2 / 2),
                          y1 + h1 / 2 - (y2 + h2 / 2)))


def normalized_center_distance(box1, box2):
    """Center distance normalised by the diagonal of the ground-truth box."""
    x2, y2, w2, h2 = box2
    diag = np.hypot(w2, h2)
    return center_distance(box1, box2) / diag if diag > 0 else None


def coverage(pred, gt):
    """Fraction of the GT box covered by the prediction (recall-like)."""
    x1, y1, w1, h1 = pred
    x2, y2, w2, h2 = gt
    xi = max(x1, x2)
    yi = max(y1, y2)
    xr = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)
    if xr <= xi or yb <= yi:
        return 0.0
    inter = (xr - xi) * (yb - yi)
    gt_area = w2 * h2
    return inter / gt_area if gt_area > 0 else 0.0


def compute_metrics(pred_roi, gt_roi):
    """Return a dict of comparison metrics between predicted and GT boxes."""
    px, py, pw, ph = pred_roi
    gx, gy, gw, gh = gt_roi
    return {
        'iou': iou(pred_roi, gt_roi),
        'center_distance_px': center_distance(pred_roi, gt_roi),
        'norm_center_distance': normalized_center_distance(pred_roi, gt_roi),
        'gt_coverage': coverage(pred_roi, gt_roi),
        'width_ratio': pw / gw if gw > 0 else None,
        'height_ratio': ph / gh if gh > 0 else None,
        'width_diff_px': abs(pw - gw),
        'height_diff_px': abs(ph - gh),
        'area_ratio': (pw * ph) / (gw * gh) if (gw * gh) > 0 else None,
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_visualization(frame, gt_roi, pred_roi, video_path, output_dir, extra_label=""):
    """
    Overlay GT (green) and predicted (red) ROIs on a frame and save as PNG.
    """
    vis = frame.copy()
    gx, gy, gw, gh = [int(v) for v in gt_roi]
    px, py, pw, ph = [int(v) for v in pred_roi]

    cv2.rectangle(vis, (gx, gy), (gx + gw, gy + gh), (0, 200, 0), 3)
    cv2.putText(vis, "GT", (gx, max(gy - 12, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2)

    cv2.rectangle(vis, (px, py), (px + pw, py + ph), (0, 0, 220), 3)
    cv2.putText(vis, "Pred", (px, max(py - 12, 50)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 220), 2)

    score = iou(pred_roi, gt_roi)
    label = f"IoU={score:.3f}  {extra_label}"
    cv2.putText(vis, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    out_path = output_dir / f"{Path(video_path).stem}_roi.png"
    cv2.imwrite(str(out_path), vis)
    return str(out_path)


# ---------------------------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------

def process_single_video(video_path, analyzer, target_frame=300):
    """
    Buffer frames up to `target_frame`, run the full detection/aggregation/
    localization chain via BreathingAnalyzer, and return the predicted ROI.

    Returns
    -------
    dict with keys:
        'roi'        : [x, y, w, h] (int) if successful, else absent
        'error'      : str if failed
        'frame'      : np.ndarray — the target frame (for visualization)
        'elapsed_s'  : float — wall-clock seconds for detection+localization
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {'error': f'Cannot open video: {video_path}'}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if target_frame >= total:
        cap.release()
        return {'error': f'target_frame={target_frame} >= total_frames={total}'}

    buffer_size = analyzer.buffer_frames_size
    start_frame = max(0, target_frame - buffer_size)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Fill the analyzer's buffer
    n_to_read = target_frame - start_frame + 1
    for _ in range(n_to_read):
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return {'error': 'Failed to read frames'}
        analyzer.buffer_frames.append(frame)

    cap.release()

    if len(analyzer.buffer_frames) < buffer_size:
        return {'error': f'Buffer only has {len(analyzer.buffer_frames)}/{buffer_size} frames'}

    target_bgr = analyzer.buffer_frames[-1].copy()

    # --- Detection + aggregation + localization via pipeline_v2 ---
    t0 = time.perf_counter()

    try:
        roi = analyzer._locate_bird_roi()
        elapsed = time.perf_counter() - t0

        return {
            'roi': [int(v) for v in roi],
            'frame': target_bgr,
            'elapsed_s': elapsed,
        }

    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {
            'error': str(e),
            'frame': target_bgr,
            'elapsed_s': elapsed,
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_summary(results, config_label):
    ok = [r for r in results if r.get('success')]
    fail = [r for r in results if not r.get('success')]
    total = len(results)

    print()
    print("=" * 70)
    print(f"SUMMARY  [{config_label}]")
    print("=" * 70)
    print(f"  Videos total   : {total}")
    print(f"  Successful      : {len(ok)} ({len(ok)/total*100:.1f}%)")
    print(f"  Failed          : {len(fail)} ({len(fail)/total*100:.1f}%)")

    if ok:
        ious = [r['iou'] for r in ok]
        dists = [r['center_distance_px'] for r in ok]
        covs = [r['gt_coverage'] for r in ok]
        print()
        print("  IoU")
        print(f"    mean   {np.mean(ious):.4f}   median {np.median(ious):.4f}"
              f"   std {np.std(ious):.4f}   min {np.min(ious):.4f}   max {np.max(ious):.4f}")
        print("  Center distance (px)")
        print(f"    mean   {np.mean(dists):.1f}   median {np.median(dists):.1f}"
              f"   std {np.std(dists):.1f}")
        print("  GT coverage (recall-like)")
        print(f"    mean   {np.mean(covs):.4f}   median {np.median(covs):.4f}")
        print()
        for thr in [0.3, 0.5, 0.7]:
            n = sum(1 for v in ious if v >= thr)
            print(f"  IoU >= {thr:.1f} : {n}/{len(ious)} ({n/len(ious)*100:.1f}%)")
    print("=" * 70)


def write_csv(results, path, config_label):
    fields = [
        'config', 'video_name', 'video_path',
        'frame_number', 'ground_truth_bpm', 'fps', 'total_frames',
        'gt_x', 'gt_y', 'gt_w', 'gt_h',
        'pred_x', 'pred_y', 'pred_w', 'pred_h',
        'iou', 'center_distance_px', 'norm_center_distance',
        'gt_coverage', 'width_ratio', 'height_ratio',
        'width_diff_px', 'height_diff_px', 'area_ratio',
        'n_hands', 'n_birds', 'elapsed_s',
        'success', 'error',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in results:
            r['config'] = config_label
            w.writerow(r)
    print(f"  CSV saved  → {path}")


def write_json(results, path, config_label, config):
    payload = {
        'config_label': config_label,
        'generated': datetime.now().isoformat(),
        'detection_mode': config.get('detection', {}).get('mode'),
        'detection_model': config.get('detection', {}).get('model'),
        'segmentation_mode': config.get('segmentation', {}).get('mode'),
        'segmentation_model': config.get('segmentation', {}).get('model'),
        'localization_method': config.get('localization', {}).get('method'),
        'results': results,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON saved → {path}")


def save_run_metadata(output_dir, config_path, args, start_time):
    """
    Save metadata about this evaluation run (timestamp, command, config used).

    Args:
        output_dir: Directory to save metadata
        config_path: Path to config file used
        args: argparse.Namespace with command-line arguments
        start_time: datetime when the run started
    """
    metadata = {
        'timestamp': start_time.isoformat(),
        'command': ' '.join(sys.argv),
        'config_file': str(config_path),
        'manifest': args.manifest,
        'mode': 'single_video' if args.video else 'all_videos',
    }

    if args.video:
        metadata['video_path'] = args.video

    # Save metadata as JSON
    metadata_path = output_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata saved → {metadata_path}")

    # Copy the config YAML to the output directory
    config_copy_path = output_dir / f'{config_path.stem}.yaml'
    shutil.copy(config_path, config_copy_path)
    print(f"  Config saved  → {config_copy_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate ROI localization against roi_manifest.json"
    )
    p.add_argument('--config', default='configs/default.yaml',
                   help='YAML config file (determines detector combo)')
    p.add_argument('--manifest', default='rois/roi_manifest.json',
                   help='Path to roi_manifest.json')
    p.add_argument('--video', default=None,
                   help='Process a single video (path). Overrides --all.')
    p.add_argument('--all', action='store_true',
                   help='Process all videos in the manifest')
    p.add_argument('--output-dir', default='rois/eval_v2',
                   help='Root output directory')
    p.add_argument('--no-vis', action='store_true',
                   help='Skip saving per-video visualizations')
    p.add_argument('--label', default=None,
                   help='Optional label override for the run '
                        '(defaults to config file stem)')
    return p.parse_args()


def main():
    args = parse_args()
    start_time = datetime.now()

    # --- Config ---
    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    run_label = args.label or config_path.stem

    # --- Output dirs ---
    out_root = Path(args.output_dir) / run_label
    plots_dir = out_root / 'plots'
    if not args.no_vis:
        plots_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    # --- Save run metadata and config ---
    save_run_metadata(out_root, config_path, args, start_time)

    # --- Manifest ---
    manifest_path = Path(args.manifest)
    with open(manifest_path) as f:
        manifest = json.load(f)

    # --- Select videos ---
    if args.video:
        video_keys = [args.video]
    elif args.all:
        video_keys = list(manifest.keys())
    else:
        print("Specify --video <path> or --all")
        sys.exit(1)

    # --- Initialize BreathingAnalyzer (loads all models) ---
    print(f"\nConfig    : {config_path}")
    print(f"Run label : {run_label}")
    print(f"Videos    : {len(video_keys)}")
    print()
    print("Initializing BreathingAnalyzer...")
    analyzer = BreathingAnalyzer(config_path=str(config_path))
    print()

    # --- Process ---
    results = []

    for idx, vpath in enumerate(video_keys, 1):
        vname = Path(vpath).name
        print(f"[{idx}/{len(video_keys)}] {vname}")

        # Ground truth
        gt_entry = manifest.get(vpath, {})
        if not gt_entry:
            print(f"  WARNING: not in manifest, skipping")
            continue

        gt_roi = gt_entry['roi']
        target_frame = gt_entry.get('frame_number', 300)
        gt_bpm = gt_entry.get('ground_truth_bpm')
        fps = gt_entry.get('metadata', {}).get('fps')
        total_frames = gt_entry.get('metadata', {}).get('total_frames')

        if not Path(vpath).exists():
            print(f"  WARNING: file not found")
            results.append({
                'video_path': vpath, 'video_name': vname,
                'frame_number': target_frame,
                'ground_truth_bpm': gt_bpm, 'fps': fps, 'total_frames': total_frames,
                'gt_x': gt_roi[0], 'gt_y': gt_roi[1],
                'gt_w': gt_roi[2], 'gt_h': gt_roi[3],
                'success': False, 'error': 'File not found',
            })
            continue

        # Clear the analyzer's buffer before each video
        analyzer.buffer_frames.clear()

        pred = process_single_video(
            video_path=vpath,
            analyzer=analyzer,
            target_frame=target_frame,
        )

        base_row = {
            'video_path': vpath,
            'video_name': vname,
            'frame_number': target_frame,
            'ground_truth_bpm': gt_bpm,
            'fps': fps,
            'total_frames': total_frames,
            'gt_x': gt_roi[0], 'gt_y': gt_roi[1],
            'gt_w': gt_roi[2], 'gt_h': gt_roi[3],
            'n_hands': 0,  # Not tracked in new version
            'n_birds': 0,  # Not tracked in new version
            'elapsed_s': pred.get('elapsed_s', 0.0),
        }

        if 'roi' in pred:
            pr = pred['roi']
            metrics = compute_metrics(pr, gt_roi)
            print(f"  IoU={metrics['iou']:.3f}  "
                  f"dist={metrics['center_distance_px']:.1f}px  "
                  f"cov={metrics['gt_coverage']:.3f}  "
                  f"t={pred['elapsed_s']:.1f}s")

            row = {
                **base_row,
                'pred_x': pr[0], 'pred_y': pr[1],
                'pred_w': pr[2], 'pred_h': pr[3],
                **metrics,
                'success': True,
                'error': '',
            }

            # Visualization
            if not args.no_vis and pred.get('frame') is not None:
                det_mode = config.get('detection', {}).get('mode', 'auto')
                seg_mode = config.get('segmentation', {}).get('mode', 'auto')
                save_visualization(pred['frame'], gt_roi, pr, vpath, plots_dir,
                                   extra_label=f"det={det_mode} seg={seg_mode}")
        else:
            err = pred.get('error', 'unknown')
            print(f"  FAILED: {err}")
            row = {
                **base_row,
                'pred_x': None, 'pred_y': None, 'pred_w': None, 'pred_h': None,
                'iou': None, 'center_distance_px': None,
                'norm_center_distance': None, 'gt_coverage': None,
                'width_ratio': None, 'height_ratio': None,
                'width_diff_px': None, 'height_diff_px': None, 'area_ratio': None,
                'success': False,
                'error': err,
            }

        results.append(row)

    # --- Output ---
    print_summary(results, run_label)

    csv_path = out_root / 'results.csv'
    json_path = out_root / 'results.json'
    write_csv(results, csv_path, run_label)
    write_json(results, json_path, run_label, config)

    print(f"\nVisualizations → {plots_dir}/")
    print(f"Done.")


if __name__ == '__main__':
    main()
