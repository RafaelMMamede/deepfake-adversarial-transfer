import json
import re
from collections import Counter

def infer_method_from_video_key(video_key: str) -> str:
    if "__" in video_key:
        return video_key.split("__", 1)[0].strip().lower()
    m = re.match(r"([a-zA-Z0-9_-]+)", video_key)
    return (m.group(1).lower() if m else "unknown")

def infer_method_from_frame_path(p: str) -> str:
    p = p.replace("\\", "/")
    parts = p.strip("/").split("/")
    if "frames" in parts:
        i = parts.index("frames")
        if i - 1 >= 0:
            return parts[i - 1].strip().lower()
    return "unknown"

def is_fake_label(v) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip().lower() in {"fake", "dfd_fake", "ff-df", "ff-f2f", "ff-fs", "ff-nt", "ff-sh",
                                     "celebdfv1_fake", "celebdfv2_fake", "dfdc_fake", "df_fake",
                                     "uadfv_fake", "roop_fake", "dfdcp_fakea", "dfdcp_fakeb"}
    if isinstance(v, int):
        return v == 1
    return False

def iter_video_entries(dataset_obj):
    """
    Yield tuples: (branch_hint, video_key, info_dict)
      - branch_hint is a string like 'fake'/'real'/None if we can infer from nesting
    Supports both:
      A) obj['fake']['train'][video_key] = info
      B) obj[split][mode][video_key] = info
    """
    # If it clearly has 'fake'/'real' branches, prioritize that structure.
    if isinstance(dataset_obj, dict) and ("fake" in dataset_obj or "real" in dataset_obj):
        for branch in ("fake", "real"):
            if branch not in dataset_obj:
                continue
            branch_dict = dataset_obj[branch]
            if not isinstance(branch_dict, dict):
                continue
            # branch_dict might be {'train': {...}, 'test': {...}} etc.
            for mode, mode_dict in branch_dict.items():
                if not isinstance(mode_dict, dict):
                    continue
                for video_key, info in mode_dict.items():
                    if isinstance(info, dict):
                        yield branch, video_key, info
        return

    # Otherwise assume obj[split][mode][video_key]
    if isinstance(dataset_obj, dict):
        for split_name, split_dict in dataset_obj.items():
            if not isinstance(split_dict, dict):
                continue
            for mode, mode_dict in split_dict.items():
                if not isinstance(mode_dict, dict):
                    continue
                for video_key, info in mode_dict.items():
                    if isinstance(info, dict):
                        yield None, video_key, info

def scan_json_methods_only_fakes(json_path: str):
    with open(json_path, "r") as f:
        j = json.load(f)

    if not isinstance(j, dict) or len(j) == 0:
        raise ValueError("JSON root must be a non-empty dict.")

    if len(j) != 1:
        print(f"[WARN] JSON has {len(j)} top-level keys: {list(j.keys())[:10]}")
    dataset_name = list(j.keys())[0]
    dataset_obj = j[dataset_name]

    by_key = Counter()
    by_path = Counter()
    mismatches = Counter()
    n_fake = 0

    for branch_hint, video_key, info in iter_video_entries(dataset_obj):
        label = info.get("label", None)

        # Decide if this entry is fake
        is_fake = False
        if branch_hint is not None:
            is_fake = (branch_hint == "fake")
        else:
            is_fake = is_fake_label(label)

        if not is_fake:
            continue

        n_fake += 1
        frames = info.get("frames", []) or []
        mk = infer_method_from_video_key(video_key)
        by_key[mk] += 1

        if frames:
            mp = infer_method_from_frame_path(frames[0])
            by_path[mp] += 1
            if mk != mp and mk != "unknown" and mp != "unknown":
                mismatches[(mk, mp)] += 1
        else:
            by_path["<no_frames>"] += 1

    return dataset_name, n_fake, by_key, by_path, mismatches

def pretty_print_counter(title, c: Counter, top=None):
    print(f"\n=== {title} (unique={len(c)}) ===")
    items = c.most_common(top) if top else c.most_common()
    for k, v in items:
        print(f"{k:20s}  {v}")

if __name__ == "__main__":
    json_path = "./dataset_json/DF40_train_bin.json"  # <-- change me

    dataset_name, n_fake, by_key, by_path, mismatches = scan_json_methods_only_fakes(json_path)

    print(f"Dataset: {dataset_name}")
    print(f"Fake entries counted: {n_fake}")

    pretty_print_counter("FAKE methods inferred from video_key prefix", by_key)
    pretty_print_counter("FAKE methods inferred from frame path folder", by_path)

    if mismatches:
        print("\n=== FAKE MISMATCHES (video_key_method -> path_method) ===")
        for (mk, mp), n in mismatches.most_common():
            print(f"{mk:20s} -> {mp:20s}  ({n})")
    else:
        print("\nNo mismatches found ✅")

    key_set = set(by_key.keys())
    path_set = set(by_path.keys())

    print("\n=== SET COMPARISONS (FAKES ONLY) ===")
    print("In key but not in path:", sorted(key_set - path_set))
    print("In path but not in key:", sorted(path_set - key_set))
