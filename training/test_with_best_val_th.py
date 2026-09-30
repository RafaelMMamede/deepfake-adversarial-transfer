import os
import json
import yaml
import argparse
import random
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from sklearn.metrics import roc_auc_score, average_precision_score


parser = argparse.ArgumentParser(
    description="Evaluate model, choose best threshold on val, and test with that threshold"
)

parser.add_argument("--detector_path", type=str, required=True, help="Path to detector YAML")
parser.add_argument("--eval_config", type=str, required=True, help="Path to eval JSON with val/test splits")
parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
parser.add_argument("--save_dir", type=str, required=True, help="Directory to save results")

parser.add_argument(
    "--dataset_root_rgb",
    type=str,
    default="./datasets",
    help="Dataset root path",
)
parser.add_argument(
    "--dataset_json_folder",
    type=str,
    default="./preprocessing/dataset_json",
    help="Folder containing dataset json files",
)

parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--workers", type=int, default=8)

parser.add_argument(
    "--threshold_metric",
    type=str,
    default="balanced_acc",
    choices=["acc", "f1", "balanced_acc"],
    help="Metric used to pick best threshold on val",
)
parser.add_argument(
    "--num_thresholds",
    type=int,
    default=1001,
    help="Number of thresholds to sweep in [0, 1]",
)

parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing JSON files even if old metrics differ from new metrics.",
)

parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Seed used for deterministic evaluation ordering.",
)

args = parser.parse_args()


# --------------------------------------------------
# IO helpers
# --------------------------------------------------
def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_json(obj, path):
    path = os.path.abspath(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"[SAVE] Wrote: {path}")

def backup_existing_file(path):
    """
    Create a timestamped backup of an existing file before overwriting it.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.backup_{timestamp}"

    shutil.copy2(path, backup_path)
    print(f"[BACKUP] Existing file backed up to: {backup_path}")

    return backup_path


def load_eval_config(path):
    cfg = load_json(path)

    if "val" not in cfg:
        raise ValueError("Eval config must contain 'val'.")

    if "test" in cfg and not isinstance(cfg["test"], dict):
        raise ValueError("Eval config 'test' must be a dict of named splits.")

    return cfg


# --------------------------------------------------
# Determinism helpers
# --------------------------------------------------
def init_seed(config):
    if config.get("manualSeed", None) is None:
        config["manualSeed"] = args.seed

    seed = int(config["manualSeed"])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    print(f"Using seed: {seed}")


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# --------------------------------------------------
# Metric comparison helpers
# --------------------------------------------------
def almost_equal(a, b, atol=1e-8, rtol=1e-6):
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return a == b

    if np.isnan(a) and np.isnan(b):
        return True

    return abs(a - b) <= atol + rtol * abs(b)


def compare_metric_dicts(
    old,
    new,
    keys=None,
    metric_atol=1e-4,
    metric_rtol=1e-4,
    count_tol=3,
):
    """
    Compare stable / already-existing metrics.

    Missing auc/ap in old files is ignored because keys not present in old
    are skipped.

    If confusion-matrix counts differ only within count_tol, derived metric
    differences are ignored because they are expected consequences of the
    count-level drift.
    """
    if keys is None:
        keys = [
            "threshold",
            "acc",
            "precision",
            "recall",
            "f1",
            "balanced_acc",
            "tp",
            "tn",
            "fp",
            "fn",
        ]

    count_keys = {"tp", "tn", "fp", "fn"}
    derived_metric_keys = {"acc", "precision", "recall", "f1", "balanced_acc"}

    differences = []

    # --------------------------------------------------
    # First compare counts
    # --------------------------------------------------
    count_diffs = []
    counts_within_tolerance = True

    for key in count_keys:
        if key not in old or key not in new:
            continue

        try:
            old_count = int(old[key])
            new_count = int(new[key])
        except Exception:
            if old[key] != new[key]:
                count_diffs.append({
                    "key": key,
                    "old": old[key],
                    "new": new[key],
                })
                counts_within_tolerance = False
            continue

        if abs(old_count - new_count) > count_tol:
            count_diffs.append({
                "key": key,
                "old": old[key],
                "new": new[key],
            })
            counts_within_tolerance = False

    differences.extend(count_diffs)

    # --------------------------------------------------
    # Then compare non-count metrics
    # --------------------------------------------------
    for key in keys:
        if key not in old or key not in new:
            continue

        if key in count_keys:
            continue

        # If counts are effectively the same, ignore derived metric drift.
        if counts_within_tolerance and key in derived_metric_keys:
            continue

        if not almost_equal(
            old[key],
            new[key],
            atol=metric_atol,
            rtol=metric_rtol,
        ):
            differences.append({
                "key": key,
                "old": old[key],
                "new": new[key],
            })

    return differences


def compare_eval_results(old_results, new_results, atol=1e-4, rtol=1e-4):
    """
    Compare existing eval_results.json with newly computed results.
    Returns list of differences.

    Missing auc/ap in old files is allowed.
    """
    differences = []

    # Compare metadata
    metadata_keys = [
        "checkpoint",
        "detector_config",
        "eval_config",
        "threshold_metric",
    ]

    for key in metadata_keys:
        old_val = old_results.get(key)
        new_val = new_results.get(key)

        if old_val != new_val:
            differences.append({
                "section": "metadata",
                "split": None,
                "key": key,
                "old": old_val,
                "new": new_val,
            })

    # Compare val metrics
    if "val" in old_results and "val" in new_results:
        val_diffs = compare_metric_dicts(
            old_results["val"],
            new_results["val"],
            metric_atol=atol,
            metric_rtol=rtol,
            count_tol=1,
        )

        for d in val_diffs:
            differences.append({
                "section": "val",
                "split": "val",
                **d,
            })

    elif "val" in old_results or "val" in new_results:
        differences.append({
            "section": "val",
            "split": "val",
            "key": "presence",
            "old": "val" in old_results,
            "new": "val" in new_results,
        })

    # Compare test split names
    old_test = old_results.get("test", {})
    new_test = new_results.get("test", {})

    old_splits = set(old_test.keys())
    new_splits = set(new_test.keys())

    if old_splits != new_splits:
        differences.append({
            "section": "test",
            "split": None,
            "key": "splits",
            "old": sorted(old_splits),
            "new": sorted(new_splits),
        })

    # Compare test metrics for common splits
    for split_name in sorted(old_splits & new_splits):
        split_diffs = compare_metric_dicts(
            old_test[split_name],
            new_test[split_name],
            metric_atol=atol,
            metric_rtol=rtol,
            count_tol=1,
        )

        for d in split_diffs:
            differences.append({
                "section": "test",
                "split": split_name,
                **d,
            })

    return differences


def print_differences(differences, max_items=20):
    print("[WARNING] Existing metrics differ from newly computed metrics.")
    print(f"Showing first {min(len(differences), max_items)} difference(s):")

    for d in differences[:max_items]:
        print(
            f"  section={d.get('section')}, "
            f"split={d.get('split')}, "
            f"key={d.get('key')}, "
            f"old={d.get('old')}, "
            f"new={d.get('new')}"
        )


def guarded_save_metric_dict(new_metrics, path, overwrite=False):
    """
    Guarded save for best_threshold.json.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"[SAVE] No existing file. Writing: {path}")
        save_json(new_metrics, path)
        return

    print(f"[SAVE] Existing file found: {path}")

    with open(path, "r") as f:
        old_metrics = json.load(f)

    differences = compare_metric_dicts(
        old_metrics,
        new_metrics,
        metric_atol=1e-4,
        metric_rtol=1e-4,
        count_tol=1,
    )

    if len(differences) == 0:
        print("[SAVE] Existing threshold metrics match new metrics. Updating file.")
        save_json(new_metrics, path)
        return

    print("[WARNING] Existing threshold metrics differ from newly computed metrics.")
    for d in differences[:20]:
        print(f"  key={d['key']}, old={d['old']}, new={d['new']}")

    if overwrite:
        print("[SAVE] --overwrite passed. Backing up and overwriting despite differences.")
        backup_existing_file(path)
        save_json(new_metrics, path)
    else:
        raise RuntimeError(
            f"Refusing to overwrite existing threshold file because metrics differ: {path}\n"
            f"Use --overwrite to force overwrite."
        )


def guarded_save_eval_results(results, path, overwrite=False):
    """
    Guarded save for eval_results.json.
    """
    path = os.path.abspath(path)

    if not os.path.exists(path):
        print(f"[SAVE] No existing file. Writing: {path}")
        save_json(results, path)
        return

    print(f"[SAVE] Existing file found: {path}")

    with open(path, "r") as f:
        old_results = json.load(f)

    differences = compare_eval_results(old_results, results)

    if len(differences) == 0:
        print("[SAVE] Existing metrics match new metrics. Updating file.")
        save_json(results, path)
        return

    print_differences(differences)

    if overwrite:
        print("[SAVE] --overwrite passed. Backing up and overwriting despite differences.")
        backup_existing_file(path)
        save_json(results, path)
    else:
        raise RuntimeError(
            f"Refusing to overwrite existing results because metrics differ: {path}\n"
            f"Use --overwrite to force overwrite."
        )


# --------------------------------------------------
# Dataset helpers
# --------------------------------------------------
def build_dataset(config, mode, split_cfg):
    local_config = dict(config)
    local_config["active_split"] = split_cfg
    return DeepfakeAbstractBaseDataset(config=local_config, mode=mode)


def make_loader(dataset, batch_size, workers):
    generator = torch.Generator()
    generator.manual_seed(args.seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=int(workers),
        collate_fn=dataset.collate_fn,
        shuffle=False,
        prefetch_factor=2 if int(workers) > 0 else None,
        worker_init_fn=seed_worker if int(workers) > 0 else None,
        generator=generator,
    )


def prepare_validation_data(config):
    val_set = build_dataset(config, mode="val", split_cfg=config["val_split"])
    return make_loader(
        dataset=val_set,
        batch_size=config["test_batchSize"],
        workers=config["workers"],
    )


def prepare_test_data(config):
    test_loaders = {}

    for split_name, split_cfg in config.get("test_splits", {}).items():
        test_set = build_dataset(config, mode="test", split_cfg=split_cfg)

        test_loaders[split_name] = make_loader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            workers=config["workers"],
        )

    return test_loaders


# --------------------------------------------------
# Model helpers
# --------------------------------------------------
def build_model(config, ckpt_path, device):
    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    ckpt = torch.load(ckpt_path, map_location="cpu")

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

    missing, unexpected = model.load_state_dict(cleaned, strict=False)

    print(f"Loaded checkpoint: {ckpt_path}")

    if missing:
        print(f"Missing keys: {len(missing)}")

    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")

    model.to(device)
    model.eval()

    return model


# --------------------------------------------------
# Evaluation helpers
# --------------------------------------------------
@torch.no_grad()
def collect_predictions(model, loader, device):
    probs_all = []
    labels_all = []

    model.eval()

    for data_dict in loader:
        if "label_spe" in data_dict:
            data_dict.pop("label_spe")

        # Match trainer behavior: convert any non-zero label to fake=1
        data_dict["label"] = torch.where(data_dict["label"] != 0, 1, 0)

        for key in data_dict.keys():
            if data_dict[key] is not None and key != "name":
                data_dict[key] = data_dict[key].to(device, non_blocking=True)

        predictions = model(data_dict, inference=True)

        if "prob" not in predictions:
            raise KeyError(
                f"Model predictions do not contain 'prob'. "
                f"Keys: {list(predictions.keys())}"
            )

        probs = predictions["prob"]
        labels = data_dict["label"]

        probs_all.append(probs.detach().cpu().numpy())
        labels_all.append(labels.detach().cpu().numpy())

    probs_all = np.concatenate(probs_all).astype(np.float64)
    labels_all = np.concatenate(labels_all).astype(np.int64)

    return probs_all, labels_all


def compute_metrics(y_true, probs, threshold):
    y_pred = (probs >= threshold).astype(np.int64)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    tnr = tn / max(tn + fp, 1)
    balanced_acc = 0.5 * (recall + tnr)

    # Threshold-independent metrics
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, probs)
        ap = average_precision_score(y_true, probs)
    else:
        auc = float("nan")
        ap = float("nan")

    return {
        "threshold": float(threshold),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_acc": float(balanced_acc),
        "auc": float(auc),
        "ap": float(ap),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def find_best_threshold(y_true, probs, metric="balanced_acc", num_thresholds=1001):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    best = None

    for th in thresholds:
        res = compute_metrics(y_true, probs, th)

        if best is None or res[metric] > best[metric]:
            best = res
        elif res[metric] == best[metric]:
            # Tie-breaker: prefer threshold closer to 0.5
            if abs(th - 0.5) < abs(best["threshold"] - 0.5):
                best = res

    return best


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    os.makedirs(args.save_dir, exist_ok=True)

    print("Running script:", os.path.abspath(__file__))
    print("Save dir:", os.path.abspath(args.save_dir))
    print("Checkpoint:", os.path.abspath(args.ckpt))
    print("Detector config:", os.path.abspath(args.detector_path))
    print("Eval config:", os.path.abspath(args.eval_config))

    detector_cfg = load_yaml(args.detector_path)
    eval_cfg = load_eval_config(args.eval_config)

    config = {}
    config.update(detector_cfg)

    # Required dataset / runtime keys
    config["ddp"] = False
    config["lmdb"] = False
    config["mode"] = "test"

    config["dataset_root_rgb"] = args.dataset_root_rgb
    config["dataset_json_folder"] = args.dataset_json_folder

    config["test_batchSize"] = args.batch_size
    config["workers"] = args.workers

    # Safe default if not present in detector config
    config.setdefault("label_dict", {"real": 0, "fake": 1})

    config["val_split"] = eval_cfg["val"]
    config["test_splits"] = eval_cfg.get("test", {})

    init_seed(config)

    # Force deterministic eval behavior
    cudnn.benchmark = False
    cudnn.deterministic = True

    print("Resolution:", config.get("resolution"))
    print("Mean:", config.get("mean"))
    print("Std:", config.get("std"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_loader = prepare_validation_data(config)
    test_loaders = prepare_test_data(config)

    model = build_model(config, args.ckpt, device)

    print("Running validation...")
    val_probs, val_labels = collect_predictions(model, val_loader, device)

    best_val = find_best_threshold(
        y_true=val_labels,
        probs=val_probs,
        metric=args.threshold_metric,
        num_thresholds=args.num_thresholds,
    )

    threshold = best_val["threshold"]

    print(f"Best val threshold: {threshold:.6f}")
    print(f"Best val {args.threshold_metric}: {best_val[args.threshold_metric]:.6f}")
    print(f"Val AUC: {best_val['auc']:.6f}")
    print(f"Val AP: {best_val['ap']:.6f}")

    results = {
        "checkpoint": args.ckpt,
        "detector_config": args.detector_path,
        "eval_config": args.eval_config,
        "threshold_metric": args.threshold_metric,
        "val": best_val,
        "test": {},
    }

    for split_name, loader in test_loaders.items():
        print(f"Evaluating test split: {split_name}")

        probs, labels = collect_predictions(model, loader, device)
        split_metrics = compute_metrics(labels, probs, threshold)

        results["test"][split_name] = split_metrics

        print(
            f"  {split_name}: "
            f"acc={split_metrics['acc']:.6f}, "
            f"balanced_acc={split_metrics['balanced_acc']:.6f}, "
            f"auc={split_metrics['auc']:.6f}, "
            f"ap={split_metrics['ap']:.6f}"
        )

    best_threshold_path = os.path.join(args.save_dir, "best_threshold.json")
    eval_results_path = os.path.join(args.save_dir, "eval_results.json")

    guarded_save_metric_dict(
        best_val,
        best_threshold_path,
        overwrite=args.overwrite,
    )

    guarded_save_eval_results(
        results,
        eval_results_path,
        overwrite=args.overwrite,
    )

    print(f"Finished. Results saved/checked in: {os.path.abspath(args.save_dir)}")


if __name__ == "__main__":
    main()