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
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from training.detectors import DETECTOR


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


# ---------------------------------------------------------------------
# Threshold loading
# ---------------------------------------------------------------------

def find_threshold_json(checkpoint_path):
    """
    Search upward from checkpoint folder for a threshold JSON.

    Expected examples:
      .../val/val/ckpt_best.pth
      .../val/val/best_threshold.json
      .../val/best_threshold.json
      .../best_threshold.json
    """
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
            "Expected one of: best_threshold.json, bestth.json, threshold.json"
        )

    with open(threshold_path, "r") as f:
        data = json.load(f)

    if "threshold" not in data:
        raise ValueError(f"No 'threshold' field found in {threshold_path}")

    threshold = float(data["threshold"])

    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Invalid threshold {threshold} from {threshold_path}")

    print("=" * 80)
    print("LOADED THRESHOLD")
    print("=" * 80)
    print(f"Threshold path: {threshold_path}")
    print(f"Threshold:      {threshold}")
    print(json.dumps(data, indent=2))

    return threshold, str(threshold_path), data


def threshold_to_tag(threshold):
    return f"{float(threshold):.8g}".replace(".", "p").replace("-", "m")


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------

def build_detector_from_checkpoint(train_config: dict, checkpoint_path: str, device):
    model_class = DETECTOR[train_config["model_name"]]
    model = model_class(train_config)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "model" in ckpt:
        state_dict = ckpt["model"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    msg = model.load_state_dict(state_dict, strict=False)
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    model.to(device)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Threshold-aware wrapper
# ---------------------------------------------------------------------

class ThresholdAwareWrappedDetector(nn.Module):
    """
    Attack expects:
      - input x in [0, 1]
      - model(x) returns logits [B, 2]

    This wrapper returns logits [0, margin], where class 1 wins iff:

        native p(fake) >= threshold

    For one-logit binary output z:
        p(fake) = sigmoid(z)
        margin = z - logit(threshold)

    For two-logit output [z_real, z_fake]:
        p(fake) = softmax([z_real, z_fake])_fake
        margin = (z_fake - z_real) - logit(threshold)

    Therefore, attacking label 1 pushes margin below 0, exactly matching
    the detector's validation-threshold decision boundary.
    """

    def __init__(self, detector, mean, std, threshold=0.5):
        super().__init__()

        self.detector = detector
        self.threshold = float(threshold)

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

    def _extract_native_output(self, x):
        x_norm = self._normalize(x)
        out = self.detector({"image": x_norm})

        if isinstance(out, dict):
            for key in ["logits", "cls", "output", "pred"]:
                if key in out:
                    out = out[key]
                    break
            else:
                if "prob" in out:
                    prob = out["prob"]

                    if torch.is_tensor(prob):
                        prob = prob.to(device=x.device, dtype=torch.float32)
                    else:
                        prob = torch.tensor(prob, device=x.device, dtype=torch.float32)

                    prob = torch.clamp(prob, 1e-6, 1.0 - 1e-6)

                    if prob.ndim == 2 and prob.shape[1] == 2:
                        prob_fake = prob[:, 1]
                    else:
                        prob_fake = prob.reshape(-1)

                    out = torch.log(prob_fake / (1.0 - prob_fake)).unsqueeze(1)
                else:
                    raise ValueError(
                        f"Could not extract logits/prob from dict keys: {list(out.keys())}"
                    )

        if not torch.is_tensor(out):
            raise TypeError(f"Model output is not a tensor: {type(out)}")

        if out.ndim == 1:
            out = out.unsqueeze(1)

        if out.ndim != 2:
            raise ValueError(f"Expected [B], [B,1], or [B,2], got {tuple(out.shape)}")

        if out.shape[1] not in [1, 2]:
            raise ValueError(f"Expected one or two logits, got {tuple(out.shape)}")

        return out

    def native_prob_fake_and_margin(self, x):
        out = self._extract_native_output(x)

        if out.shape[1] == 1:
            z = out[:, 0]
            native_prob_fake = torch.sigmoid(z)
            threshold_margin = z - self.threshold_logit

        else:
            z_real = out[:, 0]
            z_fake = out[:, 1]
            native_prob_fake = torch.softmax(out, dim=1)[:, 1]
            threshold_margin = (z_fake - z_real) - self.threshold_logit

        native_pred = (threshold_margin >= 0).long()

        return native_prob_fake, native_pred, threshold_margin

    def forward(self, x):
        _, _, threshold_margin = self.native_prob_fake_and_margin(x)

        logits = torch.stack(
            [
                torch.zeros_like(threshold_margin),
                threshold_margin,
            ],
            dim=1,
        )

        return logits


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

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


def project_delta(delta, eps, norm):
    if norm == "Linf":
        return delta.clamp(-eps, eps)

    if norm == "L2":
        b = delta.shape[0]
        flat = delta.view(b, -1)
        norm_val = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)

        factor = torch.minimum(
            torch.ones_like(norm_val),
            torch.tensor(eps, device=delta.device, dtype=delta.dtype) / norm_val,
        )

        projected = flat * factor
        return projected.view_as(delta)

    raise ValueError(f"Unsupported norm: {norm}")


def perturbation_norm(delta, norm):
    b = delta.shape[0]
    flat = delta.view(b, -1)

    if norm == "Linf":
        return flat.abs().max(dim=1).values

    if norm == "L2":
        return flat.norm(p=2, dim=1)

    raise ValueError(f"Unsupported norm: {norm}")


def attack_score_from_logits(logits, y, targeted=False, target_label=None):
    """
    Lower score is better.

    Since the threshold-aware wrapper returns logits [0, margin]:

      margin < 0  -> predicted real
      margin >= 0 -> predicted fake

    For normal fake-label attacks with y=1, lower margin is better.
    """
    margin = logits[:, 1] - logits[:, 0]

    if targeted:
        if target_label is None:
            raise ValueError("target_label must be provided for targeted attack")

        if int(target_label) == 0:
            return margin          # lower margin -> real
        elif int(target_label) == 1:
            return -margin         # higher margin -> fake
        else:
            raise ValueError(f"Invalid binary target_label: {target_label}")

    # Untargeted binary attack:
    # if true label is fake, push toward real -> lower margin
    # if true label is real, push toward fake -> higher margin
    return torch.where(y.eq(1), margin, -margin)


# ---------------------------------------------------------------------
# EoT transforms
# ---------------------------------------------------------------------

def random_affine_eot(
    x,
    degrees=0.0,
    translate=0.0,
    scale_min=1.0,
    scale_max=1.0,
    padding_mode="zeros",
):
    if degrees == 0.0 and translate == 0.0 and scale_min == 1.0 and scale_max == 1.0:
        return x

    b, c, h, w = x.shape
    device = x.device
    dtype = x.dtype

    if degrees > 0:
        angles = torch.empty(b, device=device, dtype=dtype).uniform_(
            -degrees,
            degrees,
        )
        angles = angles * math.pi / 180.0
    else:
        angles = torch.zeros(b, device=device, dtype=dtype)

    if scale_min != 1.0 or scale_max != 1.0:
        scales = torch.empty(b, device=device, dtype=dtype).uniform_(
            scale_min,
            scale_max,
        )
    else:
        scales = torch.ones(b, device=device, dtype=dtype)

    if translate > 0:
        tx = torch.empty(b, device=device, dtype=dtype).uniform_(
            -translate,
            translate,
        ) * 2.0
        ty = torch.empty(b, device=device, dtype=dtype).uniform_(
            -translate,
            translate,
        ) * 2.0
    else:
        tx = torch.zeros(b, device=device, dtype=dtype)
        ty = torch.zeros(b, device=device, dtype=dtype)

    cos_a = torch.cos(angles) * scales
    sin_a = torch.sin(angles) * scales

    theta = torch.zeros(b, 2, 3, device=device, dtype=dtype)
    theta[:, 0, 0] = cos_a
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty

    grid = F.affine_grid(theta, size=x.size(), align_corners=False)

    x_t = F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=False,
    )

    return x_t


def apply_eot_transform(
    x,
    degrees=0.0,
    translate=0.0,
    scale_min=1.0,
    scale_max=1.0,
    noise_std=0.0,
):
    x_t = random_affine_eot(
        x,
        degrees=degrees,
        translate=translate,
        scale_min=scale_min,
        scale_max=scale_max,
    )

    if noise_std > 0:
        x_t = x_t + torch.randn_like(x_t) * noise_std

    return x_t.clamp(0, 1)


# ---------------------------------------------------------------------
# C&W margin loss
# ---------------------------------------------------------------------

def cw_margin_loss(logits, y, kappa=0.0, targeted=False, target_label=None):
    """
    Untargeted C&W loss:
        max(z_y - max_{j != y} z_j + kappa, 0)

    With threshold-aware logits [0, margin] and y=1:
        max(margin + kappa, 0)

    Minimizing this pushes margin < -kappa, i.e.,
    p(fake) below the detector threshold with margin.
    """
    num_classes = logits.shape[1]

    if targeted:
        if target_label is None:
            raise ValueError("target_label must be provided for targeted attack")

        target = torch.full_like(y, int(target_label))
        target_logits = logits.gather(1, target.view(-1, 1)).squeeze(1)

        mask = F.one_hot(target, num_classes=num_classes).bool()
        other_logits = logits.masked_fill(mask, -1e9)
        max_other = other_logits.max(dim=1).values

        loss = torch.clamp(max_other - target_logits + kappa, min=0.0)
        return loss

    true_logits = logits.gather(1, y.view(-1, 1)).squeeze(1)

    mask = F.one_hot(y, num_classes=num_classes).bool()
    other_logits = logits.masked_fill(mask, -1e9)
    max_other = other_logits.max(dim=1).values

    loss = torch.clamp(true_logits - max_other + kappa, min=0.0)

    return loss


# ---------------------------------------------------------------------
# C&W + EoT attack
# ---------------------------------------------------------------------

def cw_eot_attack(
    model,
    x,
    y,
    eps=8 / 255,
    norm="Linf",
    steps=100,
    lr=1e-2,
    eot_samples=10,
    kappa=0.0,
    c=1.0,
    dist_weight=0.0,
    targeted=False,
    target_label=None,
    eot_degrees=0.0,
    eot_translate=0.0,
    eot_scale_min=1.0,
    eot_scale_max=1.0,
    eot_noise_std=0.0,
    eval_every=10,
):
    """
    Bounded C&W-margin attack with EoT.

    This version is threshold-aware because model(x) returns logits [0, margin],
    where class 1 wins iff native p(fake) >= detector threshold.

    For fixed-epsilon attacks, we keep the best image by attack score, not by
    smallest successful norm.
    """
    model.eval()

    x_orig = x.detach()
    y = y.detach()

    delta = torch.zeros_like(x_orig, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=lr)

    best_adv = x_orig.detach().clone()
    best_score = torch.full(
        (x_orig.shape[0],),
        float("inf"),
        dtype=x_orig.dtype,
        device=x_orig.device,
    )

    for step in range(steps):
        adv = (x_orig + delta).clamp(0, 1)

        eot_loss = 0.0

        for _ in range(eot_samples):
            adv_t = apply_eot_transform(
                adv,
                degrees=eot_degrees,
                translate=eot_translate,
                scale_min=eot_scale_min,
                scale_max=eot_scale_max,
                noise_std=eot_noise_std,
            )

            logits_t = model(adv_t)

            loss_vec = cw_margin_loss(
                logits_t,
                y,
                kappa=kappa,
                targeted=targeted,
                target_label=target_label,
            )

            eot_loss = eot_loss + loss_vec

        eot_loss = eot_loss / float(eot_samples)

        delta_proj = project_delta(adv - x_orig, eps=eps, norm=norm)
        dist = perturbation_norm(delta_proj, norm=norm)

        loss = c * eot_loss.mean() + dist_weight * dist.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data = project_delta(delta.data, eps=eps, norm=norm)
            delta.data = (x_orig + delta.data).clamp(0, 1) - x_orig

        should_eval = (
            (step + 1) % eval_every == 0
            or step == steps - 1
            or step == 0
        )

        if should_eval:
            with torch.no_grad():
                adv_eval = (x_orig + delta).clamp(0, 1)
                logits_eval = model(adv_eval)

                score = attack_score_from_logits(
                    logits_eval,
                    y,
                    targeted=targeted,
                    target_label=target_label,
                )

                improve = score < best_score

                best_adv[improve] = adv_eval[improve].detach()
                best_score[improve] = score[improve].detach()

    return best_adv.detach()


# ---------------------------------------------------------------------
# Reload saved images
# ---------------------------------------------------------------------

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


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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
    parser.add_argument("--label", type=int, default=1, help="Ground-truth label. For fake images use 1.")

    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--allow_threshold_default",
        action="store_true",
        help="If set, use --threshold_default when no threshold JSON is found.",
    )
    parser.add_argument("--threshold_default", type=float, default=0.5)

    # C&W / optimization params
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--kappa", type=float, default=0.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--dist_weight", type=float, default=0.0)
    parser.add_argument("--eval_every", type=int, default=10)

    # EoT params
    parser.add_argument("--eot_samples", type=int, default=10)
    parser.add_argument("--eot_degrees", type=float, default=0.0)
    parser.add_argument("--eot_translate", type=float, default=0.0)
    parser.add_argument("--eot_scale_min", type=float, default=1.0)
    parser.add_argument("--eot_scale_max", type=float, default=1.0)
    parser.add_argument("--eot_noise_std", type=float, default=0.0)

    # Targeted option
    parser.add_argument("--targeted", action="store_true")
    parser.add_argument("--target_label", type=int, default=0)

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)

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
        f"cw_eot_thaware_{args.norm.lower()}"
        f"_eps{eps_to_tag(args.eps)}"
        f"_steps{args.steps}"
        f"_eot{args.eot_samples}"
        f"_th{threshold_to_tag(threshold)}"
    )

    if args.eot_degrees > 0:
        attack_name += f"_rot{str(args.eot_degrees).replace('.', 'p')}"

    if args.eot_translate > 0:
        attack_name += f"_trans{str(args.eot_translate).replace('.', 'p')}"

    if args.eot_noise_std > 0:
        attack_name += f"_noise{str(args.eot_noise_std).replace('.', 'p')}"

    if args.targeted:
        attack_name += f"_target{args.target_label}"

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

    config_to_save = {
        "input_dir": args.input_dir,
        "checkpoint": args.checkpoint,
        "train_config": args.train_config,
        "model_tag": args.model_tag,
        "label": args.label,
        "norm": args.norm,
        "eps": args.eps,
        "threshold": threshold,
        "threshold_path": threshold_path,
        "threshold_info": threshold_info,
        "threshold_logit": model.threshold_logit,
        "steps": args.steps,
        "lr": args.lr,
        "kappa": args.kappa,
        "c": args.c,
        "dist_weight": args.dist_weight,
        "eval_every": args.eval_every,
        "eot_samples": args.eot_samples,
        "eot_degrees": args.eot_degrees,
        "eot_translate": args.eot_translate,
        "eot_scale_min": args.eot_scale_min,
        "eot_scale_max": args.eot_scale_max,
        "eot_noise_std": args.eot_noise_std,
        "targeted": args.targeted,
        "target_label": args.target_label,
        "batch_size": args.batch_size,
        "input_size": input_size,
        "mean": mean,
        "std": std,
        "out_adv_dir": str(adv_dir),
    }

    with open(results_dir / f"{attack_name}_config.json", "w") as f:
        json.dump(config_to_save, f, indent=2)

    print("=" * 80)
    print("THRESHOLD-AWARE CW-EOT")
    print("=" * 80)
    print(json.dumps(config_to_save, indent=2))

    rows = []

    for batch in tqdm(loader, desc=f"Running {attack_name} for {args.model_tag}"):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.no_grad():
            clean_prob_fake, clean_pred, clean_margin = model.native_prob_fake_and_margin(x)

        x_adv = cw_eot_attack(
            model=model,
            x=x,
            y=y,
            eps=args.eps,
            norm=args.norm,
            steps=args.steps,
            lr=args.lr,
            eot_samples=args.eot_samples,
            kappa=args.kappa,
            c=args.c,
            dist_weight=args.dist_weight,
            targeted=args.targeted,
            target_label=args.target_label,
            eot_degrees=args.eot_degrees,
            eot_translate=args.eot_translate,
            eot_scale_min=args.eot_scale_min,
            eot_scale_max=args.eot_scale_max,
            eot_noise_std=args.eot_noise_std,
            eval_every=args.eval_every,
        )

        with torch.no_grad():
            adv_prob_fake, adv_pred, adv_margin = model.native_prob_fake_and_margin(x_adv)

        initially_correct = clean_pred.eq(y)

        if args.targeted:
            attack_success = adv_pred.eq(args.target_label)
        else:
            attack_success = adv_pred.ne(y)

        perturb = x_adv - x
        perturb_norm = perturbation_norm(perturb, norm=args.norm)

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

                "perturb_norm": float(perturb_norm[i].item()),
                "targeted": int(args.targeted),
                "target_label": int(args.target_label) if args.targeted else None,
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
        "attack_name": attack_name,
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
        "steps": args.steps,
        "lr": args.lr,
        "kappa": args.kappa,
        "c": args.c,
        "dist_weight": args.dist_weight,
        "eval_every": args.eval_every,

        "eot_samples": args.eot_samples,
        "eot_degrees": args.eot_degrees,
        "eot_translate": args.eot_translate,
        "eot_scale_min": args.eot_scale_min,
        "eot_scale_max": args.eot_scale_max,
        "eot_noise_std": args.eot_noise_std,

        "targeted": args.targeted,
        "target_label": args.target_label if args.targeted else None,

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