#!/usr/bin/env python3
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a fake-only attack subset JSON from DF40-style test JSON files."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input JSON files."
    )
    parser.add_argument(
        "--out-json",
        required=True,
        help="Output attack dataset JSON path."
    )
    parser.add_argument(
        "--out-csv",
        required=True,
        help="Output manifest CSV path."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=1,
        help="How many frames to sample per clip."
    )
    parser.add_argument(
        "--subset-name",
        type=str,
        default="DF40_ATTACK_FAKE_MAIN_2000",
        help="Top-level dataset name for output JSON."
    )
    return parser.parse_args()


def infer_family_subgroup(dataset_key: str) -> Tuple[str, str]:
    """
    Example:
      DF40_TEST_EFS_other -> (EFS, other)
      DF40_TEST_FR_cdf    -> (FR, cdf)
    """
    m = re.match(r".*?_TEST_(FS|FR|EFS|FE)_(.+)$", dataset_key)
    if not m:
        raise ValueError(f"Could not infer family/subgroup from dataset key: {dataset_key}")
    return m.group(1), m.group(2)


def load_candidates(json_paths: List[Path]) -> Dict[Tuple[str, str], List[dict]]:
    """
    Returns:
      candidates[(family, subgroup)] = [
        {
          'dataset_key': ...,
          'clip_id': ...,
          'frame_path': ...,
          'label': 'fake'
        },
        ...
      ]
    """
    candidates = defaultdict(list)

    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for dataset_key, dataset_blob in data.items():
            if "fake" not in dataset_blob:
                continue
            fake_blob = dataset_blob["fake"]
            if "test" not in fake_blob:
                continue
            test_blob = fake_blob["test"]

            family, subgroup = infer_family_subgroup(dataset_key)

            for clip_id, clip_info in test_blob.items():
                if not isinstance(clip_info, dict):
                    continue

                frames = clip_info.get("frames", [])
                label = clip_info.get("label", "fake")

                for frame_path in frames:
                    candidates[(family, subgroup)].append(
                        {
                            "dataset_key": dataset_key,
                            "family": family,
                            "subgroup": subgroup,
                            "clip_id": clip_id,
                            "frame_path": frame_path,
                            "label": label,
                        }
                    )

    return candidates


def group_by_clip(records: List[dict]) -> Dict[str, List[dict]]:
    grouped = defaultdict(list)
    for r in records:
        grouped[r["clip_id"]].append(r)
    return grouped


def sample_from_group(
    records: List[dict],
    total_needed: int,
    frames_per_clip: int,
    rng: random.Random
) -> List[dict]:
    """
    Sample clip-first:
      1) shuffle clips
      2) take up to frames_per_clip frames per clip
      3) continue until total_needed reached
    """
    by_clip = group_by_clip(records)
    clip_ids = list(by_clip.keys())
    rng.shuffle(clip_ids)

    selected = []
    for clip_id in clip_ids:
        clip_frames = by_clip[clip_id][:]
        rng.shuffle(clip_frames)
        take = min(frames_per_clip, len(clip_frames), total_needed - len(selected))
        selected.extend(clip_frames[:take])
        if len(selected) >= total_needed:
            break

    return selected


def deduplicate_by_path(records: List[dict]) -> List[dict]:
    seen = set()
    out = []
    for r in records:
        p = r["frame_path"]
        if p not in seen:
            seen.add(p)
            out.append(r)
    return out


def default_quota_plan_2000():
    """
    Quotas summing to 2000.
    Family-balanced with softer subgroup handling.

    Based on earlier reasoning:
      FS:  200 cdf + 200 ff + 100 other = 500
      FR:  200 cdf + 200 ff + 100 other = 500
      EFS: 200 cdf + 200 ff + 100 other = 500
      FE:   50 cdf + 250 ff + 200 other = 500
    """
    return {
        ("FS", "cdf"): 200,
        ("FS", "ff"): 200,
        ("FS", "other"): 100,

        ("FR", "cdf"): 200,
        ("FR", "ff"): 200,
        ("FR", "other"): 100,

        ("EFS", "cdf"): 200,
        ("EFS", "ff"): 200,
        ("EFS", "other"): 100,

        ("FE", "cdf"): 50,
        ("FE", "ff"): 250,
        ("FE", "other"): 200,
    }


def rebalance_shortfalls(
    selected_by_group: Dict[Tuple[str, str], List[dict]],
    candidates: Dict[Tuple[str, str], List[dict]],
    requested_quota: Dict[Tuple[str, str], int],
    frames_per_clip: int,
    rng: random.Random
) -> Dict[Tuple[str, str], List[dict]]:
    """
    If a group cannot satisfy its requested quota, redistribute within the same family first.
    """
    families = ["FS", "FR", "EFS", "FE"]

    for family in families:
        family_groups = [k for k in requested_quota if k[0] == family]

        requested_total = sum(requested_quota[k] for k in family_groups)
        current_total = sum(len(selected_by_group.get(k, [])) for k in family_groups)
        deficit = requested_total - current_total

        if deficit <= 0:
            continue

        spares = []
        for k in family_groups:
            already_selected_paths = {r["frame_path"] for r in selected_by_group.get(k, [])}
            remaining = [r for r in deduplicate_by_path(candidates.get(k, []))
                         if r["frame_path"] not in already_selected_paths]
            if remaining:
                spares.append((k, remaining))

        if not spares:
            continue

        idx = 0
        while deficit > 0 and spares:
            k, remaining = spares[idx % len(spares)]
            already_selected_paths = {r["frame_path"] for r in selected_by_group.get(k, [])}
            remaining = [r for r in remaining if r["frame_path"] not in already_selected_paths]

            if not remaining:
                spares = [(kk, rr) for kk, rr in spares if kk != k]
                if not spares:
                    break
                continue

            extra = sample_from_group(
                remaining,
                total_needed=1,
                frames_per_clip=frames_per_clip,
                rng=rng
            )
            if extra:
                selected_by_group.setdefault(k, []).extend(extra)
                deficit -= 1
            else:
                spares = [(kk, rr) for kk, rr in spares if kk != k]

            idx += 1

    return selected_by_group


def build_attack_json(subset_name: str, selected_records: List[dict]) -> dict:
    out = {
        subset_name: {
            "real": {
                "train": {},
                "val": {},
                "test": {}
            },
            "fake": {
                "train": {},
                "val": {},
                "test": {}
            }
        }
    }

    grouped = defaultdict(list)
    for r in selected_records:
        group_name = f"{r['family']}_{r['subgroup']}"
        grouped[group_name].append(r["frame_path"])

    for group_name, frames in grouped.items():
        out[subset_name]["fake"]["test"][group_name] = {
            "label": "fake",
            "frames": sorted(frames)
        }

    return out


def write_manifest_csv(csv_path: Path, selected_records: List[dict]):
    fieldnames = [
        "sample_id",
        "family",
        "subgroup",
        "dataset_key",
        "clip_id",
        "label",
        "frame_path",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, r in enumerate(selected_records, start=1):
            row = dict(r)
            row["sample_id"] = f"{i:06d}"
            writer.writerow(row)


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    input_paths = [Path(p) for p in args.inputs]
    out_json = Path(args.out_json)
    out_csv = Path(args.out_csv)

    candidates = load_candidates(input_paths)

    quota = default_quota_plan_2000()

    print("Available candidates:")
    for k in sorted(candidates.keys()):
        print(f"  {k}: {len(candidates[k])} frames")

    selected_by_group = {}

    print("\nSampling by requested quotas:")
    for k, q in quota.items():
        records = deduplicate_by_path(candidates.get(k, []))
        chosen = sample_from_group(
            records=records,
            total_needed=q,
            frames_per_clip=args.frames_per_clip,
            rng=rng
        )
        selected_by_group[k] = chosen
        print(f"  {k}: requested={q} selected={len(chosen)}")

    selected_by_group = rebalance_shortfalls(
        selected_by_group=selected_by_group,
        candidates=candidates,
        requested_quota=quota,
        frames_per_clip=args.frames_per_clip,
        rng=rng
    )

    selected_records = []
    for k in sorted(selected_by_group.keys()):
        selected_records.extend(selected_by_group[k])

    selected_records = deduplicate_by_path(selected_records)

    print("\nFinal selected counts:")
    final_counts = defaultdict(int)
    for r in selected_records:
        final_counts[(r["family"], r["subgroup"])] += 1

    for k in sorted(final_counts.keys()):
        print(f"  {k}: {final_counts[k]}")

    print(f"\nTotal selected: {len(selected_records)}")

    attack_json = build_attack_json(args.subset_name, selected_records)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(attack_json, f, indent=2)

    write_manifest_csv(out_csv, selected_records)

    print(f"\nWrote JSON: {out_json}")
    print(f"Wrote CSV : {out_csv}")


if __name__ == "__main__":
    main()