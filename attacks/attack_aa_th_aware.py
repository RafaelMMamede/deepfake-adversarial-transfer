import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from autoattack import AutoAttack
from training.detectors import DETECTOR


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ============================================================
# Threshold loading
# ============================================================

def find_threshold_json(checkpoint_path):
    ckpt = Path(checkpoint_path).expanduser().resolve()

    candidate_names = [
        "best_threshold.json",
        "bestth.json",
        "threshold.json",
    ]

    current = ckpt.parent

    for _ in range(8):
        for name in candidate_names:
            p = current / name
            if p.exists():
                return p

        if current.parent == current:
            break

        current = current.parent

    return None


def load_threshold_from_checkpoint_folder(checkpoint_path):
    threshold_path = find_threshold_json(checkpoint_path)

    if threshold_path is None:
        raise FileNotFoundError(
            f"No threshold JSON found near checkpoint: {checkpoint_path}. "
            "Expected best_threshold.json, bestth.json, or threshold.json."
        )

    with open(threshold_path, "r") as f:
        data = json.load(f)

    if "threshold" not in data:
        raise ValueError(f"No 'threshold' field found in {threshold_path}")

    threshold = float(data["threshold"])

    if not (0.0 < threshold < 1.0):
        raise ValueError(
            f"Loaded threshold must be in (0, 1), got {threshold} from {threshold_path}"
        )

    print("=" * 80)
    print("LOADED THRESHOLD")
    print("=" * 80)
    print(f"Threshold path: {threshold_path}")
    print(f"Threshold:      {threshold}")
    print("Threshold metadata:")
    print(json.dumps(data, indent=2))

    return threshold, str(threshold_path), data


# ============================================================
# Model loading
# ============================================================

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


def build_detector_from_checkpoint(train_config: dict, checkpoint_path: str, device):
    model_class = DETECTOR[train_config["model_name"]]
    model = model_class(train_config)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = clean_state_dict(ckpt)

    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    model.to(device)
    model.eval()

    return model


# ============================================================
# Dataset
# ============================================================

class AttackDataset(Dataset):
    def __init__(self, input_dir, transform=None, label=1):
        self.input_dir = Path(input_dir)
        self.transform = transform
        self.label = int(label)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")

        self.paths = sorted(
            p for p in self.input_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )

        if not self.paths:
            raise ValueError(f"No images found in {self.input_dir}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = Image.open(path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(self.label, dtype=torch.long),
            "sample_id": path.stem,
            "input_path": str(path),
        }


# ============================================================
# Threshold-aware wrapper
# ============================================================

class ThresholdAwareWrappedDetector(nn.Module):
    """
    Wrapper for threshold-aware binary attacks.

    This version uses the same probability path as evaluate_success_efficient.py:

        detector({"image": x_norm, "label": ones}, inference=True)["prob"]

    It converts that probability to a logit and builds logits [0, threshold_margin],
    so AutoAttack's class boundary matches:

        p(fake) >= threshold
    """

    def __init__(self, detector, mean, std, threshold=0.5, prob_eps=1e-7):
        super().__init__()

        self.detector = detector
        self.threshold = float(threshold)
        self.prob_eps = float(prob_eps)

        if not (0.0 < self.threshold < 1.0):
            raise ValueError(f"threshold must be in (0, 1), got {self.threshold}")

        self.threshold_logit = float(
            math.log(self.threshold / (1.0 - self.threshold))
        )

        self.register_buffer(
            "mean",
            torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1),
        )

    def _normalize(self, x):
        return (x - self.mean) / self.std

    def _extract_eval_probability_logit(self, x):
        """
        Extract evaluator fake probability and convert it to a logit.

        Returns:
          out_logit: [B, 1]
        """

        x_norm = self._normalize(x)

        data_dict = {
            "image": x_norm,
            "label": torch.ones(x_norm.size(0), dtype=torch.long, device=x_norm.device),
        }

        out = self.detector(data_dict, inference=True)

        if not isinstance(out, dict):
            raise TypeError(f"Expected dict output from detector, got {type(out)}")

        if "prob" not in out:
            raise KeyError(f"Model output does not contain 'prob'. Keys: {list(out.keys())}")

        prob = out["prob"]

        if not torch.is_tensor(prob):
            prob = torch.tensor(prob, device=x.device, dtype=torch.float32)
        else:
            prob = prob.to(device=x.device, dtype=torch.float32)

        if prob.ndim == 2 and prob.shape[1] == 2:
            prob_fake = prob[:, 1]
        else:
            prob_fake = prob.reshape(-1)

        prob_fake = torch.clamp(
            prob_fake,
            self.prob_eps,
            1.0 - self.prob_eps,
        )

        out_logit = torch.log(prob_fake / (1.0 - prob_fake)).unsqueeze(1)

        return out_logit

    def native_prob_fake_and_margin(self, x):
        """
        Returns:
          native_prob_fake: evaluator fake probability
          native_pred: threshold-based prediction
          threshold_margin: positive means fake under threshold
        """

        out = self._extract_eval_probability_logit(x)

        z = out[:, 0]
        native_prob_fake = torch.sigmoid(z)
        threshold_margin = z - self.threshold_logit
        native_pred = (threshold_margin >= 0).long()

        return native_prob_fake, native_pred, threshold_margin

    def forward(self, x):
        """
        Return two logits for AutoAttack.

        Class 1 wins iff evaluator p(fake) >= threshold.
        """

        _, _, threshold_margin = self.native_prob_fake_and_margin(x)

        logits = torch.stack(
            [
                torch.zeros_like(threshold_margin),
                threshold_margin,
            ],
            dim=1,
        )

        return logits


# ============================================================
# Helpers
# ============================================================

def save_image(x, path):
    x = x.detach().cpu().clamp(0, 1)
    transforms.ToPILImage()(x).save(path)


def eps_to_tag(eps):
    known = {
        4 / 255: "4_255",
        8 / 255: "8_255",
        16 / 255: "16_255",
    }

    for k, v in known.items():
        if abs(eps - k) < 1e-8:
            return v

    return str(eps).replace(".", "p")


def threshold_to_tag(threshold):
    return f"{float(threshold):.8g}".replace(".", "p").replace("-", "m")


def parse_attacks_to_run(s):
    attacks = [x.strip() for x in str(s).split(",") if x.strip()]

    if not attacks:
        raise ValueError("attacks_to_run cannot be empty")

    return attacks


def gradient_sanity_check(model, loader, device):
    batch = next(iter(loader))

    x = batch["image"][: min(4, batch["image"].shape[0])].to(device)
    x.requires_grad_(True)

    logits = model(x)

    if not logits.requires_grad:
        raise RuntimeError(
            "Model logits do not require grad. "
            "inference=True may be detaching the probability."
        )

    loss = logits[:, 1].sum()
    loss.backward()

    if x.grad is None:
        raise RuntimeError("No gradient reached the input.")

    grad_max = float(x.grad.detach().abs().max().item())
    grad_mean = float(x.grad.detach().abs().mean().item())

    print("=" * 80)
    print("GRADIENT SANITY CHECK")
    print("=" * 80)
    print(f"grad max:  {grad_max:.6e}")
    print(f"grad mean: {grad_mean:.6e}")

    if grad_max == 0.0:
        raise RuntimeError(
            "Input gradient is exactly zero. "
            "Attack may not work through inference=True/prob."
        )

    x.requires_grad_(False)


# ============================================================
# Saved-image re-evaluation
# ============================================================

def evaluate_saved_images(df, model, transform, device, batch_size=32):
    saved_df = df[df["adv_path"].astype(str) != ""].copy().reset_index(drop=True)

    if len(saved_df) == 0:
        df["reload_pred"] = None
        df["reload_prob_fake"] = None
        df["reload_threshold_margin"] = None
        df["reload_attack_success"] = None
        df["reload_matches_in_memory"] = None

        return df, {
            "n_saved": 0,
            "saved_asr_all": None,
            "saved_asr_initially_correct": None,
            "reload_match_rate": None,
        }

    sample_ids = []
    reload_preds = []
    reload_probs = []
    reload_margins = []
    reload_success = []
    reload_match = []

    for start in tqdm(
        range(0, len(saved_df), batch_size),
        desc="Re-evaluating saved PNGs",
    ):
        batch_df = saved_df.iloc[start:start + batch_size]

        imgs = []
        labels = []
        adv_preds_in_memory = []

        for _, row in batch_df.iterrows():
            img = Image.open(row["adv_path"]).convert("RGB")
            img = transform(img)
            imgs.append(img)

            labels.append(int(row["label"]))
            adv_preds_in_memory.append(int(row["adv_pred"]))
            sample_ids.append(str(row["sample_id"]))

        x = torch.stack(imgs, dim=0).to(device, non_blocking=True)

        with torch.no_grad():
            prob_fake, pred, margin = model.native_prob_fake_and_margin(x)

        pred = pred.cpu().tolist()
        prob_fake = prob_fake.cpu().tolist()
        margin = margin.cpu().tolist()

        for p, pf, mg, y_i, adv_mem in zip(
            pred,
            prob_fake,
            margin,
            labels,
            adv_preds_in_memory,
        ):
            reload_preds.append(int(p))
            reload_probs.append(float(pf))
            reload_margins.append(float(mg))
            reload_success.append(int(p != y_i))
            reload_match.append(int(p == adv_mem))

    reload_df = pd.DataFrame({
        "sample_id": sample_ids,
        "reload_pred": reload_preds,
        "reload_prob_fake": reload_probs,
        "reload_threshold_margin": reload_margins,
        "reload_attack_success": reload_success,
        "reload_matches_in_memory": reload_match,
    })

    df = df.merge(reload_df, on="sample_id", how="left")

    saved_mask = df["adv_path"].astype(str) != ""
    saved_ic_mask = saved_mask & (df["initially_correct"] == 1)

    summary = {
        "n_saved": int(saved_mask.sum()),
        "saved_asr_all": (
            float(df.loc[saved_mask, "reload_attack_success"].mean())
            if saved_mask.any()
            else None
        ),
        "saved_asr_initially_correct": (
            float(df.loc[saved_ic_mask, "reload_attack_success"].mean())
            if saved_ic_mask.any()
            else None
        ),
        "reload_match_rate": (
            float(df.loc[saved_mask, "reload_matches_in_memory"].mean())
            if saved_mask.any()
            else None
        ),
    }

    return df, summary


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", required=True, help="Folder with clean images to attack")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train_config", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model_tag", required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--eps", type=float, default=8 / 255)
    parser.add_argument("--norm", default="Linf", choices=["Linf", "L2"])
    parser.add_argument("--version", default="custom")
    parser.add_argument(
        "--attacks_to_run",
        default="apgd-ce,fab,square",
        help="Comma-separated AutoAttack components, e.g. apgd-ce,fab,square",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detector fake-probability threshold. If omitted, loaded from checkpoint folder.",
    )

    parser.add_argument(
        "--allow_threshold_default",
        action="store_true",
        help="If set, use --threshold_default when no threshold JSON is found.",
    )

    parser.add_argument(
        "--threshold_default",
        type=float,
        default=0.5,
        help="Fallback threshold if --allow_threshold_default is set.",
    )

    parser.add_argument(
        "--label",
        type=int,
        default=1,
        help="Ground-truth label for all input images. For fake images use 1.",
    )

    parser.add_argument(
        "--skip_grad_check",
        action="store_true",
        help="Skip input-gradient sanity check.",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

    train_config.setdefault("label_dict", {"real": 0, "fake": 1})

    input_size = int(train_config["resolution"])
    mean = train_config["mean"]
    std = train_config["std"]

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_path = None
        threshold_info = {}
    else:
        try:
            threshold, threshold_path, threshold_info = load_threshold_from_checkpoint_folder(
                args.checkpoint
            )
        except FileNotFoundError:
            if not args.allow_threshold_default:
                raise

            threshold = float(args.threshold_default)
            threshold_path = None
            threshold_info = {}

            print("=" * 80)
            print("THRESHOLD DEFAULT WARNING")
            print("=" * 80)
            print(f"No threshold JSON found. Using threshold_default={threshold}")

    if not (0.0 < threshold < 1.0):
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    attack_name = (
        f"autoattack_thaware_"
        f"{args.norm.lower()}_eps{eps_to_tag(args.eps)}_th{threshold_to_tag(threshold)}"
    )

    out_dir = Path(args.out_dir)
    adv_dir = out_dir / "adv" / args.model_tag / attack_name
    results_dir = out_dir / "attack_results" / args.model_tag

    adv_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])

    dataset = AttackDataset(
        input_dir=args.input_dir,
        transform=transform,
        label=args.label,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    detector = build_detector_from_checkpoint(
        train_config=train_config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model = ThresholdAwareWrappedDetector(
        detector=detector,
        mean=mean,
        std=std,
        threshold=threshold,
    ).to(device).eval()

    if not args.skip_grad_check:
        gradient_sanity_check(model, loader, device)

    attacker = AutoAttack(
        model,
        norm=args.norm,
        eps=args.eps,
        version=args.version,
        device=device,
    )

    attacker.attacks_to_run = parse_attacks_to_run(args.attacks_to_run)

    config_to_save = {
        "input_dir": args.input_dir,
        "checkpoint": args.checkpoint,
        "train_config": args.train_config,
        "model_tag": args.model_tag,
        "label": args.label,
        "norm": args.norm,
        "eps": args.eps,
        "version": args.version,
        "attacks_to_run": attacker.attacks_to_run,
        "threshold": threshold,
        "threshold_path": threshold_path,
        "threshold_info": threshold_info,
        "threshold_logit": model.threshold_logit,
        "batch_size": args.batch_size,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "out_adv_dir": str(adv_dir),
    }

    with open(results_dir / f"{attack_name}_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print("=" * 80)
    print("THRESHOLD-AWARE AUTOATTACK")
    print("=" * 80)
    print(json.dumps(config_to_save, indent=2))

    rows = []

    for batch in tqdm(loader, desc=f"Running {attack_name} for {args.model_tag}"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.no_grad():
            clean_prob_fake, clean_pred, clean_margin = model.native_prob_fake_and_margin(x)

        x_adv = attacker.run_standard_evaluation(x, y, bs=x.shape[0])

        with torch.no_grad():
            adv_prob_fake, adv_pred, adv_margin = model.native_prob_fake_and_margin(x_adv)

        initially_correct = clean_pred.eq(y)
        attack_success = adv_pred.ne(y)

        for i in range(x.size(0)):
            sample_id = batch["sample_id"][i]

            adv_path = adv_dir / f"{sample_id}.png"
            save_image(x_adv[i], adv_path)

            rows.append({
                "model_tag": args.model_tag,
                "sample_id": sample_id,
                "input_path": batch["input_path"][i],
                "adv_path": str(adv_path),
                "label": int(y[i].item()),

                "threshold": float(threshold),
                "threshold_logit": float(model.threshold_logit),

                "clean_pred": int(clean_pred[i].item()),
                "clean_prob_fake": float(clean_prob_fake[i].item()),
                "clean_threshold_margin": float(clean_margin[i].item()),

                "adv_pred": int(adv_pred[i].item()),
                "adv_prob_fake": float(adv_prob_fake[i].item()),
                "adv_threshold_margin": float(adv_margin[i].item()),

                "prob_drop": float(clean_prob_fake[i].item() - adv_prob_fake[i].item()),
                "margin_drop": float(clean_margin[i].item() - adv_margin[i].item()),

                "initially_correct": int(initially_correct[i].item()),
                "attack_success": int(attack_success[i].item()),
            })

    df = pd.DataFrame(rows)

    df, saved_eval_summary = evaluate_saved_images(
        df=df,
        model=model,
        transform=transform,
        device=device,
        batch_size=args.batch_size,
    )

    results_csv = results_dir / f"{attack_name}.csv"
    summary_json = results_dir / f"{attack_name}_summary.json"

    df.to_csv(results_csv, index=False)

    ic_mask = df["initially_correct"] == 1

    summary = {
        "model_tag": args.model_tag,
        "total": int(len(df)),
        "threshold": float(threshold),
        "threshold_path": threshold_path,
        "threshold_info": threshold_info,
        "threshold_logit": float(model.threshold_logit),
        "clean_acc": float(df["initially_correct"].mean()) if len(df) else 0.0,
        "asr_all": float(df["attack_success"].mean()) if len(df) else 0.0,
        "n_initially_correct": int(ic_mask.sum()),
        "asr_initially_correct": (
            float(df.loc[ic_mask, "attack_success"].mean())
            if ic_mask.any()
            else None
        ),
        "median_clean_prob_fake_initially_correct": (
            float(df.loc[ic_mask, "clean_prob_fake"].median())
            if ic_mask.any()
            else None
        ),
        "median_adv_prob_fake_initially_correct": (
            float(df.loc[ic_mask, "adv_prob_fake"].median())
            if ic_mask.any()
            else None
        ),
        "median_prob_drop_initially_correct": (
            float(df.loc[ic_mask, "prob_drop"].median())
            if ic_mask.any()
            else None
        ),
        "median_clean_margin_initially_correct": (
            float(df.loc[ic_mask, "clean_threshold_margin"].median())
            if ic_mask.any()
            else None
        ),
        "median_adv_margin_initially_correct": (
            float(df.loc[ic_mask, "adv_threshold_margin"].median())
            if ic_mask.any()
            else None
        ),
        "median_margin_drop_initially_correct": (
            float(df.loc[ic_mask, "margin_drop"].median())
            if ic_mask.any()
            else None
        ),
        "norm": args.norm,
        "eps": args.eps,
        "version": args.version,
        "attacks_to_run": attacker.attacks_to_run,
        "label": args.label,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "adv_dir": str(adv_dir),
        "results_csv": str(results_csv),
        "saved_eval": saved_eval_summary,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()