#!/usr/bin/env python3
import argparse
import glob
import os
import random
from typing import List, Tuple

import cv2
import numpy as np


def parse_size(size_str: str) -> Tuple[int, int]:
    try:
        w, h = size_str.lower().split("x")
        return int(w), int(h)
    except Exception:
        raise argparse.ArgumentTypeError("Size must be like 1920x1080")


def list_images(images_dir: str, patterns: List[str]) -> List[str]:
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(images_dir, pat)))
    # Sort “naturally” by basename then full path for stability
    return sorted(files, key=lambda p: (os.path.basename(p).lower(), p.lower()))


def letterbox(img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize with preserved aspect ratio; pad with black to target size."""
    tw, th = target_size
    ih, iw = img.shape[:2]
    scale = min(tw / iw, th / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.zeros((th, tw, 3), dtype=np.uint8)
    x0 = (tw - nw) // 2
    y0 = (th - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def build_sequence_random(n_images: int, n_shots: int, rng: random.Random) -> List[int]:
    """Random selection with replacement, but never the same index twice in a row."""
    if n_images == 1:
        # Only one image: cannot avoid adjacency; we’ll warn at runtime.
        return [0] * n_shots
    seq = []
    prev = None
    for _ in range(n_shots):
        choices = list(range(n_images))
        if prev is not None:
            choices.remove(prev)
        pick = rng.choice(choices)
        seq.append(pick)
        prev = pick
    return seq


def build_sequence_pattern(n_images: int, n_shots: int, pattern_length: int, rng: random.Random) -> List[int]:
    """Build a small shuffled pattern and repeat it; fix boundaries to ensure no adjacency."""
    if n_images == 1:
        return [0] * n_shots
    pattern_length = max(2, min(pattern_length, n_images))  # at least 2, at most n_images
    base = list(range(n_images))
    rng.shuffle(base)
    pattern = base[:pattern_length]
    seq = []
    while len(seq) < n_shots:
        chunk = pattern[:]
        rng.shuffle(chunk)
        # Fix boundary with previous end if needed
        if seq and seq[-1] == chunk[0]:
            # rotate chunk by 1
            chunk = chunk[1:] + chunk[:1]
            if seq[-1] == chunk[0]:
                # last resort reshuffle until first differs
                attempts = 0
                while seq[-1] == chunk[0] and attempts < 10:
                    rng.shuffle(chunk)
                    attempts += 1
        seq.extend(chunk)
    return seq[:n_shots]


def main():
    ap = argparse.ArgumentParser(description="Create an MP4 from images with non-adjacent repeats.")
    ap.add_argument("--images-dir", required=True, help="Folder with images")
    ap.add_argument("--output", required=True, help="Output MP4 path (e.g., video.mp4)")
    ap.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    ap.add_argument("--duration", type=float, default=60.0, help="Duration in seconds (default: 60)")
    ap.add_argument("--shot-sec", type=float, default=0.5,
                    help="Seconds each image is shown before switching (default: 0.5)")
    ap.add_argument("--size", type=parse_size, default="1920x1080", help="Video size WxH (default: 1920x1080)")
    ap.add_argument("--pattern", default="*.jpg,*.jpeg,*.png,*.bmp", help="Comma-separated globs")
    ap.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    ap.add_argument("--mode", choices=["random", "pattern"], default="random", help="Selection strategy")
    ap.add_argument("--pattern-length", type=int, default=8, help="Length of the base pattern when mode=pattern")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    patterns = [p.strip() for p in args.pattern.split(",") if p.strip()]
    images = list_images(args.images_dir, patterns)
    if not images:
        raise SystemExit(f"No images found in {args.images_dir} matching {patterns}")

    width, height = args.size
    total_frames = int(round(args.fps * args.duration))
    if total_frames <= 0:
        raise SystemExit("Total frames must be > 0. Check fps/duration.")

    hold_frames = max(1, int(round(args.fps * args.shot_sec)))
    n_shots = max(1, int(round(total_frames / hold_frames)))
    # Recompute total frames to match exact duration (last shot absorbs remainder)
    frames_before_last = (n_shots - 1) * hold_frames
    last_shot_frames = max(1, total_frames - frames_before_last)

    # Build image index sequence with non-adjacent rule
    if args.mode == "random":
        seq = build_sequence_random(len(images), n_shots, rng)
    else:
        seq = build_sequence_pattern(len(images), n_shots, args.pattern_length, rng)

    # Prepare writer with robust codec fallback
    fourccs = ["avc1", "mp4v", "MJPG"]
    writer = None
    for tag in fourccs:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(args.output, fourcc, args.fps, (width, height))
        if w.isOpened():
            writer = w
            break
    if writer is None:
        raise SystemExit("Could not open video writer; check codecs and output path.")

    try:
        # Cache letterboxed frames so repeated images are cheap
        cache = {}

        def get_frame(path):
            if path in cache:
                return cache[path]
            img = cv2.imread(path)
            if img is None:
                return None
            frame = letterbox(img, (width, height))
            cache[path] = frame
            return frame

        for i, idx in enumerate(seq):
            path = images[idx]
            frame = get_frame(path)
            if frame is None:
                print(f"Warning: could not read {path}; skipping shot.")
                continue
            repeat = hold_frames if i < n_shots - 1 else last_shot_frames
            for _ in range(repeat):
                writer.write(frame)
    finally:
        writer.release()

    if len(images) == 1:
        print(
            "Note: Only one image was found; non-adjacent repeats are impossible. The same image was used throughout.")

    print(f"Created '{args.output}'")
    print(
        f"Shots: {n_shots} (hold {hold_frames} frames; last {last_shot_frames}) | FPS: {args.fps} | Duration: {args.duration}s")
    print(f"Images available: {len(images)} | Mode: {args.mode} | Seed: {args.seed}")
    print(f"Size: {width}x{height}")


if __name__ == "__main__":
    main()
