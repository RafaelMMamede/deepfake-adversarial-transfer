#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path
from collections import Counter, defaultdict
import multiprocessing as mp

import cv2
import numpy as np


# Fast skip patterns (do NOT attempt decoding these)
SKIP_SUBSTRINGS_DEFAULT = ["__MACOSX/", "/._", "\\__MACOSX\\", "\\._"]

def iter_frame_paths(db: dict):
    """
    Yields (root_key, dataset_name_key, label_group, split, vid, frame_path_index, frame_path)
    Works with your "DF40_*_bin" JSON structure:
      { ROOT: { label_group: { split: { vid: {label, frames:[...]} } } } }
    """
    if not isinstance(db, dict) or not db:
        return
    root_key = next(iter(db.keys()))
    root = db[root_key]
    if not isinstance(root, dict):
        return

    for label_group, splits in root.items():
        if not isinstance(splits, dict):
            continue
        for split, vids in splits.items():
            if not isinstance(vids, dict):
                continue
            for vid, info in vids.items():
                if not isinstance(info, dict):
                    continue
                frames = info.get("frames", [])
                if not isinstance(frames, list):
                    continue
                for i, p in enumerate(frames):
                    yield (root_key, label_group, split, vid, i, p)

def should_skip_path(p: str, skip_substrings):
    p2 = p.replace("\\", "/")
    for s in skip_substrings:
        if s.replace("\\", "/") in p2:
            return True
    return False

def decode_check_one(args):
    """
    Worker: attempt to read bytes + cv2.imdecode.
    Returns: (ok:bool, path:str, errtag:str)
    """
    path, max_bytes, try_extensions = args
    try:
        # Some datasets may have mixed ext case; if file missing and try_extensions enabled, try alt extensions.
        candidates = [path]
        if try_extensions and not os.path.exists(path):
            base, ext = os.path.splitext(path)
            if ext:
                exts = [ext.lower(), ext.upper()]
                # common swaps
                if ext.lower() in [".jpg", ".jpeg"]:
                    exts += [".png", ".PNG"]
                elif ext.lower() == ".png":
                    exts += [".jpg", ".JPG", ".jpeg", ".JPEG"]
                for e in dict.fromkeys(exts):  # unique preserve order
                    candidates.append(base + e)

        real_path = None
        for c in candidates:
            if os.path.exists(c):
                real_path = c
                break
        if real_path is None:
            return (False, path, "missing")

        # Read bytes (limit if requested)
        # NOTE: limiting bytes is faster but can produce false negatives for some formats.
        if max_bytes and max_bytes > 0:
            with open(real_path, "rb") as f:
                data = f.read(max_bytes)
        else:
            with open(real_path, "rb") as f:
                data = f.read()

        if not data:
            return (False, path, "empty_file")

        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # triggers libpng/jpeg decode
        if img is None:
            return (False, path, "decode_failed")

        # Extra sanity: non-zero dimensions
        if img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return (False, path, "bad_shape")

        return (True, path, "ok")

    except PermissionError:
        return (False, path, "permission")
    except OSError as e:
        # Sometimes "CRC error" may appear only as stderr; imdecode may return None instead.
        return (False, path, f"oserror:{type(e).__name__}")
    except Exception as e:
        return (False, path, f"exception:{type(e).__name__}")

def clean_json_inplace(db: dict, bad_set: set):
    """Remove bad frames; drop empty videos."""
    root_key = next(iter(db.keys()))
    root = db[root_key]

    dropped_videos = 0
    removed_frames = 0

    for label_group, splits in list(root.items()):
        if not isinstance(splits, dict):
            continue
        for split, vids in list(splits.items()):
            if not isinstance(vids, dict):
                continue
            for vid, info in list(vids.items()):
                frames = info.get("frames", [])
                if not isinstance(frames, list):
                    continue
                new_frames = [p for p in frames if p not in bad_set]
                removed_frames += (len(frames) - len(new_frames))
                if len(new_frames) == 0:
                    del vids[vid]
                    dropped_videos += 1
                else:
                    info["frames"] = new_frames

    return removed_frames, dropped_videos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True, help="Input JSON(s).")
    ap.add_argument("--workers", type=int, default=8, help="Multiprocessing workers. Default 8.")
    ap.add_argument("--max_images", type=int, default=0, help="If >0, stop after this many images per JSON.")
    ap.add_argument("--sample_every", type=int, default=1, help="Check every Nth image. Default 1 (all).")
    ap.add_argument("--skip_substrings", nargs="*", default=SKIP_SUBSTRINGS_DEFAULT,
                    help="Skip paths containing these substrings (fast).")
    ap.add_argument("--max_bytes", type=int, default=0,
                    help="If >0, only read first N bytes before decoding (faster but can false-fail). Default 0 = full file.")
    ap.add_argument("--try_extensions", action="store_true",
                    help="If a file is missing, try swapping common extensions (.png<->.jpg/.jpeg).")
    ap.add_argument("--out_dir", type=str, default="preprocessing/json_verify_reports",
                    help="Output directory for reports.")
    ap.add_argument("--show_progress", action="store_true", help="Print occasional progress lines.")
    ap.add_argument("--write_clean_json", action="store_true",
                    help="Also write a cleaned JSON with bad frames removed.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for jp in args.json:
        t0 = time.time()
        jp = str(jp)
        name = Path(jp).stem
        print(f"\n=== Verifying: {jp} ===")

        with open(jp, "r", encoding="utf-8") as f:
            db = json.load(f)

        # collect paths
        paths = []
        skipped = 0
        for k, label_group, split, vid, idx, p in iter_frame_paths(db):
            if should_skip_path(p, args.skip_substrings):
                skipped += 1
                continue
            # subsample
            if args.sample_every > 1 and (len(paths) % args.sample_every) != 0:
                # This method would skew; better use an explicit counter:
                pass

        # redo with explicit counter (correct subsampling)
        paths = []
        skipped = 0
        counter = 0
        for _, _, _, _, _, p in iter_frame_paths(db):
            if should_skip_path(p, args.skip_substrings):
                skipped += 1
                continue
            counter += 1
            if args.sample_every > 1 and (counter % args.sample_every) != 0:
                continue
            paths.append(p)
            if args.max_images and len(paths) >= args.max_images:
                break

        total = len(paths)
        print(f"frames referenced (after skip/subsample/limit): {total}  | skipped_by_pattern: {skipped}")

        if total == 0:
            print("Nothing to check.")
            continue

        # multiprocessing decode
        bad = []
        counts = Counter()

        work = [(p, args.max_bytes, args.try_extensions) for p in paths]

        # Use "spawn" for safety on some clusters; if too slow, switch to fork by removing this line.
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=args.workers) as pool:
            for i, (ok, path, tag) in enumerate(pool.imap_unordered(decode_check_one, work, chunksize=64), start=1):
                counts[tag] += 1
                if not ok:
                    bad.append((path, tag))
                if args.show_progress and (i % 5000 == 0 or i == total):
                    print(f"  checked {i}/{total} | bad={len(bad)}")

        # write outputs
        bad_txt = out_dir / f"{name}.bad_paths.txt"
        rep_json = out_dir / f"{name}.report.json"

        with bad_txt.open("w", encoding="utf-8") as f:
            for p, tag in bad:
                f.write(f"{p}\t{tag}\n")

        report = {
            "input_json": jp,
            "checked_frames": total,
            "skipped_by_pattern": skipped,
            "bad_frames": len(bad),
            "counts": dict(counts),
            "elapsed_sec": round(time.time() - t0, 2),
        }
        with rep_json.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"bad_frames: {len(bad)}")
        # compact top tags
        top = counts.most_common(10)
        print("top tags:", ", ".join([f"{k}={v}" for k, v in top]))
        print(f"wrote: {bad_txt}")
        print(f"wrote: {rep_json}")

        if args.write_clean_json:
            bad_set = set(p for p, _ in bad)
            removed_frames, dropped_videos = clean_json_inplace(db, bad_set)
            out_clean = out_dir / f"{name}.clean.json"
            with out_clean.open("w", encoding="utf-8") as f:
                json.dump(db, f, indent=2)
            print(f"wrote: {out_clean} (removed_frames={removed_frames}, dropped_videos={dropped_videos})")

    print("\nDone.")

if __name__ == "__main__":
    main()
