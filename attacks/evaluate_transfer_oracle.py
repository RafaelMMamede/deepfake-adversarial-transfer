import argparse
import json
from pathlib import Path
import sys
import gc
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from training.detectors import DETECTOR
import yaml


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------
# datasets
# ---------------------------------------

class CleanImageDataset(Dataset):
    """
    Dataset for clean images only.
    """
    def __init__(self, image_dir, transform):
        self.image_dir = Path(image_dir)
        self.transform = transform

        if not self.image_dir.exists():
            raise FileNotFoundError(f"Clean image directory not found: {self.image_dir}")

        self.files = sorted([
            p.name for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        image = Image.open(self.image_dir / fname).convert("RGB")

        return {
            "image": self.transform(image),
            "id": fname,
        }


class MultiAttackImageDataset(Dataset):
    """
    Dataset over all adversarial folders at once.

    Expected attack_records entries:
        {
            "source_group": "...",      # e.g. imgnet, fr
            "origin_model": "...",      # e.g. vit_ALL
            "origin_key": "...",        # e.g. imgnet_vit_ALL
            "attack_name": "...",
            "attack_path": Path(...)
        }

    Each returned sample keeps attack metadata, so the final predictions
    can be grouped back into origin_model / attack_name.
    """
    def __init__(self, attack_records, transform, allowed_files):
        self.transform = transform
        self.allowed_files = set(allowed_files)
        self.samples = []

        for rec in attack_records:
            attack_path = Path(rec["attack_path"])
            source_group = rec.get("source_group", "")
            origin_model = rec["origin_model"]
            origin_key = rec.get("origin_key", origin_model)
            attack_name = rec["attack_name"]

            if not attack_path.exists():
                continue

            for p in sorted(attack_path.iterdir()):
                if (
                    p.is_file()
                    and p.suffix.lower() in IMAGE_EXTS
                    and p.name in self.allowed_files
                ):
                    self.samples.append({
                        "path": p,
                        "id": p.name,
                        "source_group": source_group,
                        "origin_model": origin_model,
                        "origin_key": origin_key,
                        "attack_name": attack_name,
                        "attack_path": str(attack_path),
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["path"]).convert("RGB")

        return {
            "image": self.transform(image),
            "id": item["id"],
            "source_group": item["source_group"],
            "origin_model": item["origin_model"],
            "origin_key": item["origin_key"],
            "attack_name": item["attack_name"],
            "attack_path": item["attack_path"],
        }


# ---------------------------------------
# discovery / loading utilities
# ---------------------------------------

def has_images(folder):
    folder = Path(folder)

    if not folder.is_dir():
        return False

    return any(
        p.is_file() and p.suffix.lower() in IMAGE_EXTS
        for p in folder.iterdir()
    )


def parse_adv_root_spec(spec):
    """
    Accept either:
        /path/to/adv_root
    or:
        label=/path/to/adv_root

    The label becomes source_group. This is important when different roots
    contain the same origin folder names, e.g. imgnet vit_ALL and fr vit_ALL.
    """
    spec = str(spec)

    if "=" in spec:
        source_group, adv_root = spec.split("=", 1)
        source_group = source_group.strip()
        adv_root = adv_root.strip()

        if not source_group:
            raise ValueError(f"Invalid adv root spec with empty label: {spec}")
    else:
        adv_root = spec
        source_group = Path(adv_root).name

    return source_group, Path(adv_root)


def find_attack_records(adv_root, source_group=None):
    """
    Expected structure:

        adv_root/
          origin_model/
            attack_name/
              image1.png
              image2.png

    Returns:
        list of {
            "source_group": source_group,
            "origin_model": origin_model,
            "origin_key": source_group_origin_model,
            "attack_name": attack_name,
            "attack_path": attack_dir
        }
    """
    adv_root = Path(adv_root)

    if source_group is None:
        source_group = adv_root.name

    if not adv_root.exists():
        raise FileNotFoundError(f"Adversarial root not found: {adv_root}")

    records = []

    for origin_dir in sorted([p for p in adv_root.iterdir() if p.is_dir()]):
        origin_model = origin_dir.name
        origin_key = f"{source_group}_{origin_model}" if source_group else origin_model

        for attack_dir in sorted([p for p in origin_dir.iterdir() if p.is_dir()]):
            attack_name = attack_dir.name

            if has_images(attack_dir):
                records.append({
                    "source_group": source_group,
                    "origin_model": origin_model,
                    "origin_key": origin_key,
                    "attack_name": attack_name,
                    "attack_path": attack_dir,
                })

    return records


def load_threshold_from_ckpt(ckpt_path):
    th_path = Path(ckpt_path).parent / "best_threshold.json"

    if not th_path.is_file():
        raise FileNotFoundError(f"Threshold file not found: {th_path}")

    with open(th_path, "r") as f:
        data = json.load(f)

    if "threshold" not in data:
        raise KeyError(f"'threshold' not found in {th_path}")

    return float(data["threshold"])


def clean_state_dict(ckpt):
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    cleaned = {}

    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v

    return cleaned


def load_detector(model_info, device):
    with open(model_info["config"]) as f:
        config = yaml.safe_load(f)

    config.setdefault("label_dict", {"real": 0, "fake": 1})

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    ckpt = torch.load(model_info["ckpt"], map_location=device)
    state_dict = clean_state_dict(ckpt)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    if missing:
        print(f"  missing keys: {len(missing)}")

    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    threshold = load_threshold_from_ckpt(model_info["ckpt"])
    resolution = int(config.get("resolution", 224))

    return model, config, threshold, resolution


def get_model_mean_std(config):
    """
    Read the normalization statistics used by the detector.

    The oracle evaluator must use the same preprocessing as the standard
    transfer evaluator. Silently falling back to unnormalized tensors would
    make its clean accuracy, white-box ASR, black-box ASR, and oracle ASR
    incomparable with the threshold-aware evaluation.
    """
    if "mean" not in config or "std" not in config:
        raise KeyError(
            "Detector config is missing 'mean' and/or 'std'; "
            "cannot reproduce the model's evaluation preprocessing."
        )

    mean = [float(x) for x in config["mean"]]
    std = [float(x) for x in config["std"]]

    if len(mean) != 3 or len(std) != 3:
        raise ValueError(
            "Expected three-channel normalization statistics, "
            f"but received mean={mean} and std={std}."
        )

    if any(s <= 0 for s in std):
        raise ValueError(f"All normalization std values must be positive: {std}")

    return mean, std


def make_transform(resolution, no_resize, mean, std):
    tfms = []

    if not no_resize:
        tfms.append(transforms.Resize((resolution, resolution)))

    tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return transforms.Compose(tfms)


def make_loader(dataset, batch_size, num_workers, device):
    """
    Safer DataLoader settings for large batches / cluster filesystems.

    pin_memory=False avoids the pin-memory thread crash.
    persistent_workers=False avoids keeping broken workers alive between loaders.
    """
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": False,
    }

    if num_workers > 0:
        kwargs["persistent_workers"] = False
        kwargs["prefetch_factor"] = 2

    return DataLoader(**kwargs)


def cleanup_model(model, device):
    del model
    gc.collect()

    if device == "cuda":
        torch.cuda.empty_cache()


# ---------------------------------------
# resume / skip utilities
# ---------------------------------------

def load_existing_results(output_csv, expected_num_attacks):
    """
    Load existing output CSV and determine which target models are complete.

    A target model is considered complete if it has at least expected_num_attacks rows.
    This is safer than simply checking whether the target_model appears in the CSV,
    because a crashed run may have written only partial rows.
    """
    output_path = Path(output_csv)

    results = []
    completed_targets = set()
    incomplete_targets = set()

    if not output_path.exists() or output_path.stat().st_size == 0:
        print("No existing output CSV found. Starting from scratch.")
        return results, completed_targets, incomplete_targets

    existing_df = pd.read_csv(output_path)

    if len(existing_df) == 0 or "target_model" not in existing_df.columns:
        print("Existing output CSV is empty or missing target_model. Starting from scratch.")
        return results, completed_targets, incomplete_targets

    counts = existing_df.groupby("target_model").size()

    completed_targets = set(
        counts[counts >= expected_num_attacks].index
    )

    incomplete_targets = set(counts.index) - completed_targets

    print("\nResume information:")
    print(f"  Existing rows: {len(existing_df)}")
    print(f"  Expected rows per complete target model: {expected_num_attacks}")
    print(f"  Completed target models: {len(completed_targets)}")
    print(f"  Incomplete target models: {len(incomplete_targets)}")

    for name in sorted(completed_targets):
        print(f"    completed: {name} rows={counts[name]}")

    for name in sorted(incomplete_targets):
        print(f"    incomplete: {name} rows={counts[name]}")

    results = existing_df.to_dict("records")

    return results, completed_targets, incomplete_targets


# ---------------------------------------
# prediction
# ---------------------------------------

@torch.inference_mode()
def predict_clean(model, loader, device, threshold, desc):
    rows = []
    model.eval()

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)

        data_dict = {
            "image": x,
            "label": torch.ones(x.size(0), dtype=torch.long, device=device),
        }

        out = model(data_dict, inference=True)

        if "prob" not in out:
            raise KeyError(f"Model output does not contain 'prob'. Keys: {list(out.keys())}")

        prob = out["prob"].detach().cpu().view(-1)
        pred = (prob >= threshold).long()

        for fname, p, yhat in zip(batch["id"], prob.tolist(), pred.tolist()):
            rows.append({
                "id": fname,
                "prob_fake": float(p),
                "pred": int(yhat),
            })

    return pd.DataFrame(rows)


@torch.inference_mode()
def predict_all_attacks(model, loader, device, threshold, desc):
    rows = []
    model.eval()

    for batch in tqdm(loader, desc=desc, leave=False):
        x = batch["image"].to(device, non_blocking=True)

        data_dict = {
            "image": x,
            "label": torch.ones(x.size(0), dtype=torch.long, device=device),
        }

        out = model(data_dict, inference=True)

        if "prob" not in out:
            raise KeyError(f"Model output does not contain 'prob'. Keys: {list(out.keys())}")

        prob = out["prob"].detach().cpu().view(-1)
        pred = (prob >= threshold).long()

        for fname, source_group, origin_model, origin_key, attack_name, attack_path, p, yhat in zip(
            batch["id"],
            batch["source_group"],
            batch["origin_model"],
            batch["origin_key"],
            batch["attack_name"],
            batch["attack_path"],
            prob.tolist(),
            pred.tolist(),
        ):
            rows.append({
                "id": fname,
                "source_group": source_group,
                "origin_model": origin_model,
                "origin_key": origin_key,
                "attack_name": attack_name,
                "attack_path": attack_path,
                "prob_fake": float(p),
                "pred": int(yhat),
            })

    return pd.DataFrame(rows)


# ---------------------------------------
# metrics
# ---------------------------------------

def compute_metrics_from_predictions(clean_df, adv_df):
    merged = clean_df.merge(
        adv_df,
        on="id",
        how="inner",
        suffixes=("_clean", "_adv"),
    )

    if len(merged) == 0:
        return None

    label = 1  # fake-only attack set

    clean_correct_mask = merged["pred_clean"] == label
    adv_correct_mask = merged["pred_adv"] == label
    success_mask = clean_correct_mask & (merged["pred_adv"] != label)

    # Fake-only probability drop:
    # positive value means adversarial image reduced the target model's fake probability.
    prob_drop = merged["prob_fake_clean"] - merged["prob_fake_adv"]

    total = int(len(merged))
    clean_correct = int(clean_correct_mask.sum())
    adv_correct = int(adv_correct_mask.sum())
    success = int(success_mask.sum())

    # Confidence-conditioned probability drop:
    # use the 10% images with highest clean fake probability.
    # This is threshold-independent and remains defined even when clean_correct == 0.
    top_conf_frac = 0.10
    n_top_conf = max(1, int(math.ceil(total * top_conf_frac)))

    top_conf_idx = merged["prob_fake_clean"].nlargest(n_top_conf).index
    top_conf_mask = merged.index.isin(top_conf_idx)

    avg_prob_drop_top10_conf = (
        float(prob_drop[top_conf_mask].mean())
        if top_conf_mask.any()
        else float("nan")
    )

    median_prob_drop_top10_conf = (
        float(prob_drop[top_conf_mask].median())
        if top_conf_mask.any()
        else float("nan")
    )

    # Sanity check: how many of the top-confidence images were actually clean-correct.
    # If this is low, the model is confidently wrong on the attack set.
    top10_conf_clean_acc = (
        float(clean_correct_mask[top_conf_mask].mean())
        if top_conf_mask.any()
        else float("nan")
    )

    # Main probability drop: same eligibility subset as ASR.
    # This is the metric you asked for: prob drop on previously correct images.
    avg_prob_drop_clean_correct = (
        float(prob_drop[clean_correct_mask].mean())
        if clean_correct_mask.any()
        else float("nan")
    )

    median_prob_drop_clean_correct = (
        float(prob_drop[clean_correct_mask].median())
        if clean_correct_mask.any()
        else float("nan")
    )

    # Optional diagnostic: probability drop over all samples.
    # Useful to see whether the attack moves probabilities even when the model was already wrong.
    avg_prob_drop_all = (
        float(prob_drop.mean())
        if total > 0
        else float("nan")
    )

    median_prob_drop_all = (
        float(prob_drop.median())
        if total > 0
        else float("nan")
    )

    # Probability drop only among actually successful attacks.
    avg_prob_drop_success = (
        float(prob_drop[success_mask].mean())
        if success_mask.any()
        else float("nan")
    )

    median_prob_drop_success = (
        float(prob_drop[success_mask].median())
        if success_mask.any()
        else float("nan")
    )

    return {
        "clean_acc": clean_correct / total if total > 0 else float("nan"),
        "adv_acc": adv_correct / total if total > 0 else float("nan"),

        # If no clean-correct samples exist, ASR is undefined.
        "asr": success / clean_correct if clean_correct > 0 else float("nan"),

        "avg_prob_drop_clean_correct": avg_prob_drop_clean_correct,
        "median_prob_drop_clean_correct": median_prob_drop_clean_correct,

        # Threshold-independent diagnostic on the most confident clean samples.
        "avg_prob_drop_top10_conf": avg_prob_drop_top10_conf,
        "median_prob_drop_top10_conf": median_prob_drop_top10_conf,
        "top10_conf_clean_acc": top10_conf_clean_acc,
        "num_top10_conf": int(top_conf_mask.sum()),

        # Optional diagnostic over all images.
        "avg_prob_drop_all": avg_prob_drop_all,
        "median_prob_drop_all": median_prob_drop_all,

        # Probability drop among successful attacks only.
        "avg_prob_drop_success": avg_prob_drop_success,
        "median_prob_drop_success": median_prob_drop_success,

        "num_samples": total,
        "num_clean_correct": clean_correct,
        "num_adv_correct": adv_correct,
        "num_success": success,
    }


def compute_metrics_for_all_attacks(clean_df, adv_all_df):
    rows = []

    group_cols = ["source_group", "origin_model", "origin_key", "attack_name", "attack_path"]

    for keys, adv_df in adv_all_df.groupby(group_cols):
        source_group, origin_model, origin_key, attack_name, attack_path = keys
        metrics = compute_metrics_from_predictions(clean_df, adv_df)

        if metrics is None:
            continue

        rows.append({
            "source_group": source_group,
            "origin_model": origin_model,
            "origin_key": origin_key,
            "attack_name": attack_name,
            "attack_path": attack_path,

            "num_samples": metrics["num_samples"],

            "clean_acc": metrics["clean_acc"],
            "adv_acc": metrics["adv_acc"],
            "asr": metrics["asr"],

            "avg_prob_drop_clean_correct": metrics["avg_prob_drop_clean_correct"],
            "median_prob_drop_clean_correct": metrics["median_prob_drop_clean_correct"],

            "avg_prob_drop_top10_conf": metrics["avg_prob_drop_top10_conf"],
            "median_prob_drop_top10_conf": metrics["median_prob_drop_top10_conf"],
            "top10_conf_clean_acc": metrics["top10_conf_clean_acc"],
            "num_top10_conf": metrics["num_top10_conf"],

            "avg_prob_drop_all": metrics["avg_prob_drop_all"],
            "median_prob_drop_all": metrics["median_prob_drop_all"],

            "avg_prob_drop_success": metrics["avg_prob_drop_success"],
            "median_prob_drop_success": metrics["median_prob_drop_success"],

            "num_clean_correct": metrics["num_clean_correct"],
            "num_adv_correct": metrics["num_adv_correct"],
            "num_success": metrics["num_success"],
        })

    return rows




# ---------------------------------------
# model-name parsing for semantic oracle rules
# ---------------------------------------

TRAIN_DATA_TAGS = {"ALL", "FS", "FR", "EFS", "FE"}
KNOWN_ARCHITECTURES = {
    "resnet34",
    "xception",
    "efficientnetb4",
    "deit",
    "swin",
    "vit",
    "clip",
}


def parse_model_semantics(name):
    """
    Parse names such as:
        imgnet_vit_ALL
        fr_xception_EFS
        vit_ALL
        no_pretrain_resnet34_FS

    Returns architecture and train_data when they can be inferred.
    This intentionally ignores pretraining/source_group for the two strict
    oracle rules: same architecture and same exact training data.
    """
    name = str(name)
    parts = [p for p in name.split("_") if p]

    train_data = None
    if parts and parts[-1] in TRAIN_DATA_TAGS:
        train_data = parts[-1]
        core_parts = parts[:-1]
    else:
        core_parts = parts

    architecture = None
    for part in reversed(core_parts):
        if part in KNOWN_ARCHITECTURES:
            architecture = part
            break

    # Fallback for any future naming pattern: use the last token before train tag.
    if architecture is None and core_parts:
        architecture = core_parts[-1]

    return {
        "name": name,
        "architecture": architecture,
        "train_data": train_data,
    }


def load_oracle_rules(path):
    """
    Optional JSON file for per-target exclusions.

    Example:
    {
      "default": {
        "exclude_origin_contains": ["bad_source"],
        "exclude_source_groups": ["old_no_pretrain"],
        "exclude_train_tags_as_source": ["ALL"]
      },
      "imgnet_vit_ALL": {
        "exclude_origin_keys": ["imgnet_vit_ALL"],
        "exclude_attack_name_contains": ["debug"]
      }
    }
    """
    if path is None:
        return {}

    with open(path, "r") as f:
        return json.load(f)


def get_oracle_rule(rules, target_name):
    default = rules.get("default", {})
    specific = rules.get(target_name, {})

    # Merge list-like fields by concatenation; scalar/bool fields are overridden.
    merged = dict(default)

    for k, v in specific.items():
        if isinstance(v, list) and isinstance(merged.get(k), list):
            merged[k] = merged[k] + v
        else:
            merged[k] = v

    return merged


def _contains_any(series, patterns):
    if not patterns:
        return pd.Series(False, index=series.index)

    mask = pd.Series(False, index=series.index)

    for pat in patterns:
        mask |= series.astype(str).str.contains(str(pat), regex=False, na=False)

    return mask


def filter_oracle_candidates(adv_all_df, target_name, rule, exclude_self_origin=False):
    """
    Apply target-specific rules before oracle selection.
    The goal is to define which source adversarial examples are eligible.
    """
    d = adv_all_df.copy()

    if len(d) == 0:
        return d

    # Include filters first, if provided.
    include_source_groups = rule.get("include_source_groups", [])
    if include_source_groups:
        d = d[d["source_group"].isin(include_source_groups)]

    include_origin_keys = rule.get("include_origin_keys", [])
    if include_origin_keys:
        d = d[d["origin_key"].isin(include_origin_keys)]

    include_origin_models = rule.get("include_origin_models", [])
    if include_origin_models:
        d = d[d["origin_model"].isin(include_origin_models)]

    # Exclude self-source for transfer analysis.
    if exclude_self_origin or rule.get("exclude_self_origin", False):
        d = d[
            (d["origin_key"] != target_name)
            & (d["origin_model"] != target_name)
        ]

    # Semantic strict-transfer filters. These are applied dynamically, so you
    # do not need to manually enumerate all forbidden origin/target pairs.
    target_sem = parse_model_semantics(target_name)

    # Exclude all sources trained with selected train tags, regardless of target.
    # Example: exclude_train_tags_as_source=["ALL"] removes ALL-trained
    # source attacks from the oracle pool, while still allowing ALL as a target.
    exclude_train_tags_as_source = rule.get("exclude_train_tags_as_source", [])
    if exclude_train_tags_as_source:
        exclude_train_tags_as_source = set(map(str, exclude_train_tags_as_source))
        origin_train_data = d["origin_model"].map(
            lambda x: parse_model_semantics(x)["train_data"]
        )
        d = d[~origin_train_data.isin(exclude_train_tags_as_source)]

    if rule.get("exclude_same_training_data", False):
        target_train_data = target_sem["train_data"]

        if target_train_data is None:
            print(
                f"  warning: could not infer training data for target={target_name}; "
                "same-training-data exclusion was not applied for this target."
            )
        else:
            origin_train_data = d["origin_model"].map(
                lambda x: parse_model_semantics(x)["train_data"]
            )
            d = d[origin_train_data != target_train_data]

    if rule.get("exclude_same_architecture", False):
        target_architecture = target_sem["architecture"]

        if target_architecture is None:
            print(
                f"  warning: could not infer architecture for target={target_name}; "
                "same-architecture exclusion was not applied for this target."
            )
        else:
            origin_architecture = d["origin_model"].map(
                lambda x: parse_model_semantics(x)["architecture"]
            )
            d = d[origin_architecture != target_architecture]

    exclude_source_groups = rule.get("exclude_source_groups", [])
    if exclude_source_groups:
        d = d[~d["source_group"].isin(exclude_source_groups)]

    exclude_origin_keys = rule.get("exclude_origin_keys", [])
    if exclude_origin_keys:
        d = d[~d["origin_key"].isin(exclude_origin_keys)]

    exclude_origin_models = rule.get("exclude_origin_models", [])
    if exclude_origin_models:
        d = d[~d["origin_model"].isin(exclude_origin_models)]

    exclude_attack_names = rule.get("exclude_attack_names", [])
    if exclude_attack_names:
        d = d[~d["attack_name"].isin(exclude_attack_names)]

    exclude_origin_contains = rule.get("exclude_origin_contains", [])
    if exclude_origin_contains:
        m = (
            _contains_any(d["origin_key"], exclude_origin_contains)
            | _contains_any(d["origin_model"], exclude_origin_contains)
        )
        d = d[~m]

    exclude_attack_name_contains = rule.get("exclude_attack_name_contains", [])
    if exclude_attack_name_contains:
        d = d[~_contains_any(d["attack_name"], exclude_attack_name_contains)]

    exclude_attack_path_contains = rule.get("exclude_attack_path_contains", [])
    if exclude_attack_path_contains:
        d = d[~_contains_any(d["attack_path"], exclude_attack_path_contains)]

    return d


def select_worst_adv_per_image(eligible_adv_df):
    """
    For each image id, select the adversarial example that is worst for the target.

    Since this is a fake-only attack set and fake=1, lower prob_fake is worse.
    Minimizing prob_fake also maximizes the probability drop for that image because
    the clean probability is fixed per target/id.
    """
    if len(eligible_adv_df) == 0:
        return eligible_adv_df.copy(), pd.Series(dtype=int)

    candidate_counts = eligible_adv_df.groupby("id").size()

    # Deterministic tie-breaking after prob_fake.
    d = eligible_adv_df.sort_values(
        ["id", "prob_fake", "origin_key", "attack_name", "attack_path"],
        ascending=[True, True, True, True, True],
    )

    selected = d.groupby("id", sort=False).head(1).copy()
    selected["num_candidates_for_id"] = selected["id"].map(candidate_counts).astype(int)

    return selected, candidate_counts


def compute_oracle_worst_metrics(clean_df, adv_all_df, target_name, rule, exclude_self_origin=False):
    eligible = filter_oracle_candidates(
        adv_all_df=adv_all_df,
        target_name=target_name,
        rule=rule,
        exclude_self_origin=exclude_self_origin,
    )

    selected_adv, candidate_counts = select_worst_adv_per_image(eligible)

    if len(selected_adv) == 0:
        return None, selected_adv, None

    metrics = compute_metrics_from_predictions(clean_df, selected_adv)

    if metrics is None:
        return None, selected_adv, None

    selected_counts = (
        selected_adv
        .groupby(["source_group", "origin_model", "origin_key", "attack_name", "attack_path"])
        .size()
        .reset_index(name="num_selected")
        .sort_values("num_selected", ascending=False)
    )

    row = {
        "aggregation": "oracle_worst_min_prob_fake",
        "oracle_rule_json": json.dumps(rule, sort_keys=True),
        "num_eligible_adv_rows": int(len(eligible)),
        "num_unique_images_with_candidates": int(selected_adv["id"].nunique()),
        "mean_candidates_per_image": float(candidate_counts.mean()),
        "min_candidates_per_image": int(candidate_counts.min()),
        "max_candidates_per_image": int(candidate_counts.max()),

        "num_samples": metrics["num_samples"],

        "clean_acc": metrics["clean_acc"],
        "adv_acc": metrics["adv_acc"],
        "asr": metrics["asr"],

        "avg_prob_drop_clean_correct": metrics["avg_prob_drop_clean_correct"],
        "median_prob_drop_clean_correct": metrics["median_prob_drop_clean_correct"],

        "avg_prob_drop_top10_conf": metrics["avg_prob_drop_top10_conf"],
        "median_prob_drop_top10_conf": metrics["median_prob_drop_top10_conf"],
        "top10_conf_clean_acc": metrics["top10_conf_clean_acc"],
        "num_top10_conf": metrics["num_top10_conf"],

        "avg_prob_drop_all": metrics["avg_prob_drop_all"],
        "median_prob_drop_all": metrics["median_prob_drop_all"],

        "avg_prob_drop_success": metrics["avg_prob_drop_success"],
        "median_prob_drop_success": metrics["median_prob_drop_success"],

        "num_clean_correct": metrics["num_clean_correct"],
        "num_adv_correct": metrics["num_adv_correct"],
        "num_success": metrics["num_success"],
    }

    return row, selected_adv, selected_counts


# ---------------------------------------
# main
# ---------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean_dir", required=True)

    parser.add_argument(
        "--adv_roots",
        nargs="+",
        required=True,
        help="One or more adversarial roots with structure root/origin_model/attack_name/images.",
    )

    parser.add_argument("--models_json", required=True)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_csv", default="transfer_results_single_adv_loader.csv")
    parser.add_argument(
        "--output_oracle_csv",
        default=None,
        help="CSV for multi-source oracle/worst-case transfer results. Default: <output_csv stem>_oracle.csv",
    )
    parser.add_argument(
        "--output_oracle_selection_csv",
        default=None,
        help="Optional per-image selected source CSV. Default: <output_csv stem>_oracle_selection.csv",
    )
    parser.add_argument(
        "--oracle_rules_json",
        default=None,
        help="Optional JSON with per-target source exclusions/inclusions for oracle selection.",
    )
    parser.add_argument(
        "--exclude_self_origin",
        action="store_true",
        help="For oracle transfer, exclude the source with the same origin_key/origin_model as the target.",
    )
    parser.add_argument(
        "--force_recompute",
        action="store_true",
        help="Recompute targets even if output_csv already contains complete individual rows. Useful when adding oracle outputs.",
    )

    parser.add_argument(
        "--no_resize",
        action="store_true",
        help="Skip resizing. Use this if clean and adversarial images are already at model resolution.",
    )

    parser.add_argument(
        "--attack_name_contains",
        default=None,
        help="Optional substring filter for attack names.",
    )

    parser.add_argument(
        "--origin_model_contains",
        default=None,
        help="Optional substring filter for origin model names.",
    )

    args = parser.parse_args()

    if args.output_oracle_csv is None:
        p = Path(args.output_csv)
        args.output_oracle_csv = str(p.with_name(p.stem + "_oracle" + p.suffix))

    if args.output_oracle_selection_csv is None:
        p = Path(args.output_csv)
        args.output_oracle_selection_csv = str(p.with_name(p.stem + "_oracle_selection" + p.suffix))

    oracle_rules = load_oracle_rules(args.oracle_rules_json)
    oracle_results = []
    oracle_selection_rows = []
    oracle_source_count_rows = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ---------------------------------------
    # discover attack records
    # ---------------------------------------
    attack_records = []

    for adv_root_spec in args.adv_roots:
        source_group, adv_root = parse_adv_root_spec(adv_root_spec)
        found = find_attack_records(adv_root, source_group=source_group)
        print(f"Found {len(found)} attack folders under {adv_root} [source_group={source_group}]")
        attack_records.extend(found)

    # Deduplicate by full path
    unique = {}
    for rec in attack_records:
        unique[str(rec["attack_path"])] = rec

    attack_records = list(unique.values())

    if args.attack_name_contains is not None:
        attack_records = [
            rec for rec in attack_records
            if args.attack_name_contains in rec["attack_name"]
        ]

    if args.origin_model_contains is not None:
        attack_records = [
            rec for rec in attack_records
            if (
                args.origin_model_contains in rec["origin_model"]
                or args.origin_model_contains in rec.get("origin_key", rec["origin_model"])
            )
        ]

    attack_records = sorted(
        attack_records,
        key=lambda r: (r["source_group"], r["origin_model"], r["attack_name"], str(r["attack_path"]))
    )

    print(f"Total attack folders after filtering: {len(attack_records)}")

    if len(attack_records) == 0:
        raise ValueError("No attack folders found after filtering.")

    print("\nAttack folders:")
    for rec in attack_records:
        print(
            f"  source_group={rec['source_group']} | "
            f"origin={rec['origin_model']} | "
            f"origin_key={rec['origin_key']} | "
            f"attack={rec['attack_name']} | "
            f"path={rec['attack_path']}"
        )

    # ---------------------------------------
    # load model list
    # ---------------------------------------
    with open(args.models_json) as f:
        models = json.load(f)

    # ---------------------------------------
    # load previous results / determine skip list
    # ---------------------------------------
    expected_num_attacks = len(attack_records)

    if args.force_recompute:
        print(
            "\n--force_recompute enabled: ignoring all existing individual "
            "results and recomputing every target model."
        )
        results = []
        completed_targets = set()
        incomplete_targets = set()
    else:
        results, completed_targets, incomplete_targets = load_existing_results(
            output_csv=args.output_csv,
            expected_num_attacks=expected_num_attacks,
        )

    # ---------------------------------------
    # evaluate target models
    # ---------------------------------------
    for model_info in models:
        target_name = model_info["name"]

        if target_name in completed_targets:
            print(f"\nSkipping already completed target model: {target_name}")
            continue

        if target_name in incomplete_targets:
            print(f"\nRecomputing incomplete target model: {target_name}")

            # Remove old partial rows for this target before recomputing
            results = [
                row for row in results
                if row.get("target_model") != target_name
            ]

            # Save cleaned CSV immediately
            pd.DataFrame(results).to_csv(args.output_csv, index=False)

        print("\n" + "=" * 80)
        print(f"Evaluating target model: {target_name}")
        print("=" * 80)

        model, config, threshold, resolution = load_detector(model_info, device)
        mean, std = get_model_mean_std(config)

        print(f"  threshold: {threshold:.6f}")
        print(f"  resolution: {resolution}")
        print(f"  normalization mean: {mean}")
        print(f"  normalization std: {std}")

        transform = make_transform(
            resolution=resolution,
            no_resize=args.no_resize,
            mean=mean,
            std=std,
        )

        # ---------------------------------------
        # clean predictions
        # ---------------------------------------
        clean_dataset = CleanImageDataset(
            image_dir=args.clean_dir,
            transform=transform,
        )

        if len(clean_dataset) == 0:
            raise ValueError(f"No clean images found in {args.clean_dir}")

        clean_loader = make_loader(
            dataset=clean_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        print("\nComputing clean predictions...")

        clean_df = predict_clean(
            model=model,
            loader=clean_loader,
            device=device,
            threshold=threshold,
            desc=f"Clean {target_name}",
        )

        clean_files = set(clean_df["id"].tolist())

        # ---------------------------------------
        # all adversarial predictions in one DataLoader
        # ---------------------------------------
        print("\nBuilding adversarial dataset over all attack folders...")

        adv_dataset = MultiAttackImageDataset(
            attack_records=attack_records,
            transform=transform,
            allowed_files=clean_files,
        )

        if len(adv_dataset) == 0:
            print("  skipped: no matching adversarial images found")
            cleanup_model(model, device)
            continue

        print(f"  total adversarial images to evaluate: {len(adv_dataset)}")

        adv_loader = make_loader(
            dataset=adv_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        print("\nComputing adversarial predictions...")

        adv_all_df = predict_all_attacks(
            model=model,
            loader=adv_loader,
            device=device,
            threshold=threshold,
            desc=f"Adv all attacks {target_name}",
        )

        attack_metrics = compute_metrics_for_all_attacks(
            clean_df=clean_df,
            adv_all_df=adv_all_df,
        )

        for metrics in attack_metrics:
            results.append({
                "target_model": target_name,
                "target_threshold": threshold,

                "source_group": metrics["source_group"],
                "origin_model": metrics["origin_model"],
                "origin_key": metrics["origin_key"],
                "attack_name": metrics["attack_name"],
                "attack_path": metrics["attack_path"],

                "num_samples": metrics["num_samples"],

                "clean_acc": metrics["clean_acc"],
                "adv_acc": metrics["adv_acc"],
                "asr": metrics["asr"],

                "avg_prob_drop_clean_correct": metrics["avg_prob_drop_clean_correct"],
                "median_prob_drop_clean_correct": metrics["median_prob_drop_clean_correct"],

                "avg_prob_drop_top10_conf": metrics["avg_prob_drop_top10_conf"],
                "median_prob_drop_top10_conf": metrics["median_prob_drop_top10_conf"],
                "top10_conf_clean_acc": metrics["top10_conf_clean_acc"],
                "num_top10_conf": metrics["num_top10_conf"],

                "avg_prob_drop_all": metrics["avg_prob_drop_all"],
                "median_prob_drop_all": metrics["median_prob_drop_all"],

                "avg_prob_drop_success": metrics["avg_prob_drop_success"],
                "median_prob_drop_success": metrics["median_prob_drop_success"],

                "num_clean_correct": metrics["num_clean_correct"],
                "num_adv_correct": metrics["num_adv_correct"],
                "num_success": metrics["num_success"],
            })

            asr_str = (
                f"{metrics['asr']:.4f}"
                if not pd.isna(metrics["asr"])
                else "nan"
            )

            prob_drop_str = (
                f"{metrics['avg_prob_drop_clean_correct']:.4f}"
                if not pd.isna(metrics["avg_prob_drop_clean_correct"])
                else "nan"
            )

            top10_drop_str = (
                f"{metrics['avg_prob_drop_top10_conf']:.4f}"
                if not pd.isna(metrics["avg_prob_drop_top10_conf"])
                else "nan"
            )

            print(
                f"  source_group={metrics['source_group']} | "
                f"origin={metrics['origin_model']} | "
                f"origin_key={metrics['origin_key']} | "
                f"attack={metrics['attack_name']} | "
                f"clean_acc={metrics['clean_acc']:.4f} | "
                f"adv_acc={metrics['adv_acc']:.4f} | "
                f"asr={asr_str} | "
                f"avg_prob_drop_clean_correct={prob_drop_str} | "
                f"avg_prob_drop_top10_conf={top10_drop_str}"
            )

        # ---------------------------------------
        # oracle / worst-case multi-source transfer
        # ---------------------------------------
        rule = get_oracle_rule(oracle_rules, target_name)

        oracle_row, selected_adv_df, selected_counts_df = compute_oracle_worst_metrics(
            clean_df=clean_df,
            adv_all_df=adv_all_df,
            target_name=target_name,
            rule=rule,
            exclude_self_origin=args.exclude_self_origin,
        )

        if oracle_row is None:
            print("\nOracle worst-case transfer: skipped because no eligible candidates remained.")
        else:
            oracle_row = {
                "target_model": target_name,
                "target_threshold": threshold,
                **oracle_row,
            }
            oracle_results.append(oracle_row)

            asr_str = f"{oracle_row['asr']:.4f}" if not pd.isna(oracle_row["asr"]) else "nan"
            drop_str = (
                f"{oracle_row['avg_prob_drop_clean_correct']:.4f}"
                if not pd.isna(oracle_row["avg_prob_drop_clean_correct"])
                else "nan"
            )

            print(
                "\nOracle worst-case transfer | "
                f"eligible_adv_rows={oracle_row['num_eligible_adv_rows']} | "
                f"images={oracle_row['num_unique_images_with_candidates']} | "
                f"mean_candidates/image={oracle_row['mean_candidates_per_image']:.2f} | "
                f"clean_acc={oracle_row['clean_acc']:.4f} | "
                f"adv_acc={oracle_row['adv_acc']:.4f} | "
                f"asr={asr_str} | "
                f"avg_prob_drop_clean_correct={drop_str}"
            )

            # Per-image selections can be useful for debugging which source won.
            selection_keep_cols = [
                "id", "source_group", "origin_model", "origin_key", "attack_name",
                "attack_path", "prob_fake", "pred", "num_candidates_for_id",
            ]
            tmp_sel = selected_adv_df[selection_keep_cols].copy()
            tmp_sel.insert(0, "target_model", target_name)
            tmp_sel.rename(
                columns={"prob_fake": "selected_prob_fake", "pred": "selected_pred"},
                inplace=True,
            )
            oracle_selection_rows.extend(tmp_sel.to_dict("records"))

            tmp_counts = selected_counts_df.copy()
            tmp_counts.insert(0, "target_model", target_name)
            oracle_source_count_rows.extend(tmp_counts.to_dict("records"))

        # Save partial after each target model
        pd.DataFrame(results).to_csv(args.output_csv, index=False)
        pd.DataFrame(oracle_results).to_csv(args.output_oracle_csv, index=False)
        pd.DataFrame(oracle_selection_rows).to_csv(args.output_oracle_selection_csv, index=False)

        source_counts_path = str(
            Path(args.output_oracle_csv).with_name(
                Path(args.output_oracle_csv).stem
                + "_source_counts"
                + Path(args.output_oracle_csv).suffix
            )
        )
        pd.DataFrame(oracle_source_count_rows).to_csv(source_counts_path, index=False)

        print(f"\nPartial individual results saved to: {args.output_csv}")
        print(f"Partial oracle results saved to: {args.output_oracle_csv}")
        print(f"Partial oracle selections saved to: {args.output_oracle_selection_csv}")
        print(f"Partial oracle source counts saved to: {source_counts_path}")

        cleanup_model(model, device)

    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)

    oracle_df = pd.DataFrame(oracle_results)
    oracle_df.to_csv(args.output_oracle_csv, index=False)

    oracle_selection_df = pd.DataFrame(oracle_selection_rows)
    oracle_selection_df.to_csv(args.output_oracle_selection_csv, index=False)

    source_counts_path = str(
        Path(args.output_oracle_csv).with_name(
            Path(args.output_oracle_csv).stem
            + "_source_counts"
            + Path(args.output_oracle_csv).suffix
        )
    )
    oracle_source_counts_df = pd.DataFrame(oracle_source_count_rows)
    oracle_source_counts_df.to_csv(source_counts_path, index=False)

    print("\nSaved final individual results to:")
    print(args.output_csv)
    print("Saved final oracle results to:")
    print(args.output_oracle_csv)
    print("Saved final oracle per-image selections to:")
    print(args.output_oracle_selection_csv)
    print("Saved final oracle source counts to:")
    print(source_counts_path)


if __name__ == "__main__":
    main()