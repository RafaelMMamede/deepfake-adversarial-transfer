import sys
import os
import argparse
import importlib
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, RandomSampler

import wandb
from sklearn.metrics import roc_curve, auc

# ---------------------------------------------------------------------
# Project setup
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

TRAINING_ROOT = ROOT / "training"
sys.path.append(str(TRAINING_ROOT))

from pretraining.utils import losses
from pretraining.utils.dataset import RFWVerificationDataset, RaceFaceFRPretrainDataset
from pretraining.utils.verification_aux import evaluate_verification, compute_accuracy
from pretraining.utils.utils import set_seed

from training.networks.resnet34 import ResNet34
from training.networks.xception import Xception
from training.networks.efficientnetb4 import EfficientNetB4

try:
    from detectors import DETECTOR
except Exception:
    DETECTOR = None

torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------
def load_config(config_path):
    if config_path is None:
        raise ValueError("Please provide a config file with --config")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return config_module.config


# ---------------------------------------------------------------------
# FR model wrappers
# ---------------------------------------------------------------------
class EmbeddingHead(nn.Module):
    def __init__(self, in_features, embedding_size=512, dropout=0.0):
        super().__init__()

        layers = [nn.BatchNorm1d(in_features)]

        if dropout and dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        layers.extend([
            nn.Linear(in_features, embedding_size, bias=False),
            nn.BatchNorm1d(embedding_size),
        ])

        self.head = nn.Sequential(*layers)

    def forward(self, x):
        x = self.head(x)
        return F.normalize(x, p=2, dim=1)


class FRPretrainWrapper(nn.Module):
    def __init__(self, detector_backbone, feature_dim, embedding_size=512, dropout=0.0):
        super().__init__()

        self.detector_backbone = detector_backbone
        self.embedding_head = EmbeddingHead(
            in_features=feature_dim,
            embedding_size=embedding_size,
            dropout=dropout,
        )

    def forward(self, x):
        feats = self.detector_backbone.features(x)

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1)
            feats = torch.flatten(feats, 1)

        return self.embedding_head(feats)


class DetectorClassFRPretrainWrapper(nn.Module):
    def __init__(self, detector, feature_dim, embedding_size=512, dropout=0.0):
        super().__init__()

        self.detector = detector
        self.embedding_head = EmbeddingHead(
            in_features=feature_dim,
            embedding_size=embedding_size,
            dropout=dropout,
        )

    def forward(self, x):
        if hasattr(self.detector, "features"):
            feats = self.detector.features({"image": x})
        elif hasattr(self.detector, "backbone"):
            feats = self.detector.backbone(x)
        else:
            raise AttributeError(
                "Detector has neither .features(data_dict) nor .backbone."
            )

        if isinstance(feats, (tuple, list)):
            feats = feats[0]

        if feats.ndim == 4:
            feats = F.adaptive_avg_pool2d(feats, 1)
            feats = torch.flatten(feats, 1)

        return self.embedding_head(feats)


# ---------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------
def get_feature_dim(network, mode):
    if mode == "adjust_channel":
        return 512

    feature_dims = {
        "resnet34": 512,
        "xception": 2048,
        "efficientnetb4": 1792,
    }

    if network not in feature_dims:
        raise ValueError(
            f"Unsupported explicit-backbone network: {network}. "
            "For detector-only models, use use_detector_class=True and set feature_dim."
        )

    return feature_dims[network]


def build_detector_config(cfg):
    return {
        "num_classes": getattr(cfg, "num_classes_for_dummy_head", 2),
        "inc": getattr(cfg, "inc", 3),
        "dropout": getattr(cfg, "dropout", 0.0),
        "mode": getattr(cfg, "mode", "normal"),
        "pretrained": getattr(cfg, "initial_pretrained", None),
    }


def load_detector_backbone(cfg):
    detector_config = build_detector_config(cfg)

    if cfg.network == "resnet34":
        return ResNet34(detector_config)

    if cfg.network == "xception":
        return Xception(detector_config)

    if cfg.network == "efficientnetb4":
        return EfficientNetB4(detector_config)

    raise ValueError(
        f"Unsupported network for explicit-backbone FR pretraining: {cfg.network}. "
        "Supported: resnet34, xception, efficientnetb4. "
        "For detector-only models, set use_detector_class=True."
    )


def load_detector_class(cfg):
    if DETECTOR is None:
        raise ImportError(
            "Could not import DETECTOR registry. Check that ROOT/training was added "
            "to sys.path and that `from detectors import DETECTOR` works."
        )

    detector_modules = getattr(cfg, "detector_modules", [])

    for module_name in detector_modules:
        print(f"[INFO] Importing detector module: {module_name}", flush=True)
        importlib.import_module(module_name)

    detector_name = getattr(cfg, "detector_name", cfg.network)

    try:
        detector_cls = DETECTOR[detector_name]
    except KeyError:
        available = getattr(DETECTOR, "data", None)

        if isinstance(available, dict):
            available = list(available.keys())
        else:
            available = "Could not list available detectors."

        raise ValueError(
            f"Detector '{detector_name}' is not registered in DETECTOR.\n"
            f"Available detectors: {available}\n"
            f"Make sure the detector is imported in training/detectors/__init__.py."
        )

    detector_config = getattr(cfg, "detector_config", {}).copy()

    detector_config.setdefault(
        "loss_func",
        getattr(cfg, "detector_loss_func", "cross_entropy"),
    )

    detector = detector_cls(detector_config)

    return detector


def infer_detector_feature_dim(detector, cfg):
    feature_dim = getattr(cfg, "feature_dim", None)

    if feature_dim is not None:
        return int(feature_dim)

    if hasattr(detector, "backbone") and hasattr(detector.backbone, "num_features"):
        return int(detector.backbone.num_features)

    raise ValueError(
        "Could not infer detector feature_dim. "
        "Please set `feature_dim = ...` in the config."
    )


def load_fr_model(cfg, device):
    use_detector_class = getattr(cfg, "use_detector_class", False)

    if use_detector_class:
        detector = load_detector_class(cfg)
        feature_dim = infer_detector_feature_dim(detector, cfg)

        print(
            f"[INFO] Using detector-class mode: "
            f"detector_name={getattr(cfg, 'detector_name', cfg.network)}, "
            f"feature_dim={feature_dim}",
            flush=True,
        )

        model = DetectorClassFRPretrainWrapper(
            detector=detector,
            feature_dim=feature_dim,
            embedding_size=cfg.embedding_size,
            dropout=getattr(cfg, "embedding_dropout", 0.0),
        )

        return model.to(device)

    detector_backbone = load_detector_backbone(cfg)

    feature_dim = get_feature_dim(
        network=cfg.network,
        mode=getattr(cfg, "mode", "normal"),
    )

    print(
        f"[INFO] Using explicit-backbone mode: "
        f"network={cfg.network}, feature_dim={feature_dim}",
        flush=True,
    )

    model = FRPretrainWrapper(
        detector_backbone=detector_backbone,
        feature_dim=feature_dim,
        embedding_size=cfg.embedding_size,
        dropout=getattr(cfg, "embedding_dropout", 0.0),
    )

    return model.to(device)


def load_header(cfg, device, num_classes):
    if cfg.loss == "ElasticArcFace":
        header = losses.ElasticArcFace(
            cfg.embedding_size,
            num_classes,
            cfg.s,
            cfg.m,
            cfg.std,
        )

    elif cfg.loss == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(
            cfg.embedding_size,
            num_classes,
            cfg.s,
            cfg.m,
            cfg.std,
            plus=True,
        )

    elif cfg.loss == "AdaFace":
        header = losses.AdaFace(
            cfg.embedding_size,
            num_classes,
            cfg.m,
            cfg.h,
            cfg.s,
            cfg.t_alpha,
        )

    else:
        raise ValueError(f"Invalid FR loss/header: {cfg.loss}")

    return header.to(device)


# ---------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------
def build_train_dataset(cfg):
    return RaceFaceFRPretrainDataset(
        dataset_dir=cfg.rec,
        ethnicity=getattr(cfg, "ethnicity", "All"),
        image_size=getattr(cfg, "image_size", 224),
        is_train=True,
        return_path=False,
        min_images_per_identity=getattr(cfg, "min_images_per_identity", 1),
        cache_dir=getattr(cfg, "cache_dir", "./pretraining/cache"),
        use_cache=getattr(cfg, "use_cache", True),
        rebuild_cache=getattr(cfg, "rebuild_cache", False),
    )


def build_val_dataset(cfg):
    return RFWVerificationDataset(
        dataset_dir=cfg.rfw_dir,
        image_size=getattr(cfg, "rfw_image_size", getattr(cfg, "image_size", 224)),
        ethnicity=getattr(cfg, "rfw_ethnicity", None),
        return_path=False,
    )


# ---------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------
def build_optimizer_and_scheduler(model, header, cfg, steps_per_epoch):
    scaled_lr = cfg.lr / 512 * cfg.batch_size

    optimizer_name = getattr(cfg, "optimizer", "sgd").lower()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(header.parameters()),
            lr=scaled_lr,
            weight_decay=cfg.weight_decay,
            betas=getattr(cfg, "adamw_betas", (0.9, 0.999)),
            eps=getattr(cfg, "adamw_eps", 1e-8),
        )
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            list(model.parameters()) + list(header.parameters()),
            lr=scaled_lr,
            momentum=getattr(cfg, "momentum", 0.9),
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    warmup_epochs = getattr(cfg, "warmup_epochs", 0)
    warmup_steps = int(warmup_epochs * steps_per_epoch)

    def lr_lambda(global_step):
        epoch_float = global_step / steps_per_epoch

        base_factor = cfg.lr_func(epoch_float)

        if warmup_steps > 0 and global_step < warmup_steps:
            warmup_factor = float(global_step + 1) / float(warmup_steps)
        else:
            warmup_factor = 1.0

        return warmup_factor * base_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    print(f"[INFO] Optimizer={optimizer_name}", flush=True)
    print(f"[INFO] Base cfg.lr={cfg.lr}", flush=True)
    print(f"[INFO] batch_size={cfg.batch_size}", flush=True)
    print(f"[INFO] Scaled actual LR={scaled_lr}", flush=True)
    print(f"[INFO] steps_per_epoch={steps_per_epoch}", flush=True)
    print(f"[INFO] warmup_epochs={warmup_epochs}", flush=True)
    print(f"[INFO] warmup_steps={warmup_steps}", flush=True)

    return optimizer, scheduler


# ---------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------
def save_checkpoint(path, model, header, cfg, best_val_auc, num_classes, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    ckpt = {
        "network": cfg.network,
        "mode": getattr(cfg, "mode", "normal"),
        "image_size": getattr(cfg, "image_size", None),
        "embedding_size": cfg.embedding_size,
        "best_val_auc": best_val_auc,
        "epoch": epoch,
        "num_identity_classes": num_classes,
        "embedding_head": model.embedding_head.state_dict(),
        "fr_header": header.state_dict(),
    }

    if hasattr(model, "detector_backbone"):
        ckpt["pretrain_mode"] = "backbone_class"
        ckpt["detector_backbone"] = model.detector_backbone.state_dict()

    elif hasattr(model, "detector"):
        ckpt["pretrain_mode"] = "detector_class"
        ckpt["detector"] = model.detector.state_dict()

        if hasattr(model.detector, "backbone"):
            ckpt["detector_backbone"] = model.detector.backbone.state_dict()

    else:
        raise AttributeError(
            "Model has neither .detector_backbone nor .detector. "
            "Cannot save pretrained backbone."
        )

    torch.save(ckpt, path)


# ---------------------------------------------------------------------
# WandB
# ---------------------------------------------------------------------
def make_wandb_config(cfg):
    wandb_config = {}

    for k, v in cfg.__dict__.items():
        if k.startswith("__"):
            continue

        if callable(v):
            continue

        if isinstance(v, (staticmethod, classmethod)):
            continue

        if isinstance(v, (str, int, float, bool, type(None))):
            wandb_config[k] = v
        elif isinstance(v, (list, tuple)):
            wandb_config[k] = list(v)
        elif isinstance(v, dict):
            wandb_config[k] = v
        else:
            wandb_config[k] = str(v)

    return wandb_config


def setup_wandb(cfg):
    use_wandb = getattr(cfg, "use_wandb", False)

    output_dir = Path(getattr(cfg, "output_dir", "./pretraining/checkpoints/default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not use_wandb:
        return False, output_dir

    wandb.login()

    wandb.init(
        project=getattr(cfg, "wandb_project", "fr_pretrain_deepfake_backbones"),
        name=getattr(cfg, "wandb_job_name", None),
        config=make_wandb_config(cfg),
    )

    return True, output_dir


# ---------------------------------------------------------------------
# Train / validation
# ---------------------------------------------------------------------
def train_one_epoch(
    model,
    header,
    train_loader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    device,
    cfg,
    epoch,
    global_step,
    use_wandb,
):
    model.train()
    header.train()

    print(
        f"[TRAIN] Starting epoch {epoch + 1}/{cfg.num_epoch} | "
        f"model.training={model.training} | header.training={header.training}",
        flush=True,
    )

    running_loss = 0.0
    use_amp = getattr(cfg, "use_amp", True)

    for step, (img, label) in enumerate(train_loader):
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            features = model(img)

        features = features.float()

        with torch.cuda.amp.autocast(enabled=False):
            logits = header(features, label)
            loss = criterion(logits.float(), label)

        if not torch.isfinite(loss):
            print(
                f"[WARN] Non-finite loss at epoch={epoch} step={step}: {loss.item()}",
                flush=True,
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        clip_grad_norm_(
            list(model.parameters()) + list(header.parameters()),
            max_norm=getattr(cfg, "grad_clip", 5.0),
        )

        scaler.step(optimizer)
        scaler.update()

        # Step LR scheduler every batch so warmup is smooth.
        scheduler.step()

        running_loss += loss.item()
        global_step += 1

        if step % getattr(cfg, "log_interval", 100) == 0:
            avg_loss = running_loss / max(1, step + 1)

            log_dict = {
                "train/loss": loss.item(),
                "train/avg_loss": avg_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": epoch,
                "batch_step": step,
                "global_step": global_step,
            }

            if use_wandb:
                wandb.log(log_dict)
            else:
                print(log_dict, flush=True)

    return global_step


@torch.no_grad()
def validate(model, val_loader, device):
    model.eval()

    scores, labels = evaluate_verification(model, val_loader, device)

    fpr, tpr, _ = roc_curve(labels, scores)
    val_auc = auc(fpr, tpr)

    _, val_acc = compute_accuracy(scores, labels)

    return val_auc, val_acc


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main(args):
    set_seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Device: {device}", flush=True)
    print(f"[INFO] Config: {args.config}", flush=True)

    use_wandb, output_dir = setup_wandb(cfg)

    print("[INFO] Building train dataset...", flush=True)
    train_dataset = build_train_dataset(cfg)
    print(
        f"[INFO] Train dataset built: {len(train_dataset)} images, "
        f"{train_dataset.num_classes} identities.",
        flush=True,
    )

    print("[INFO] Building validation dataset...", flush=True)
    val_dataset = build_val_dataset(cfg)
    print(f"[INFO] Validation dataset built: {len(val_dataset)} pairs.", flush=True)

    num_workers = getattr(cfg, "num_workers", 8)

    train_loader_kwargs = {
        "dataset": train_dataset,
        "batch_size": cfg.batch_size,
        "sampler": RandomSampler(train_dataset),
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": True,
    }

    if num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = getattr(cfg, "prefetch_factor", 4)
        train_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(**train_loader_kwargs)

    val_loader_kwargs = {
        "dataset": val_dataset,
        "batch_size": getattr(cfg, "val_batch_size", 256),
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
    }

    if num_workers > 0:
        val_loader_kwargs["persistent_workers"] = True

    val_loader = DataLoader(**val_loader_kwargs)

    print("[INFO] Building model...", flush=True)
    model = load_fr_model(cfg, device)


    print("[INFO] Building FR header...", flush=True)
    header = load_header(cfg, device, train_dataset.num_classes)

    criterion = CrossEntropyLoss()

    optimizer, scheduler = build_optimizer_and_scheduler(
        model=model,
        header=header,
        cfg=cfg,
        steps_per_epoch=len(train_loader),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=getattr(cfg, "use_amp", True))

    best_val_auc = -1.0
    global_step = 0

    best_path = os.path.join(output_dir, "best_fr_pretrained_detector_backbone.pth")
    last_path = os.path.join(output_dir, "last_fr_pretrained_detector_backbone.pth")

    for epoch in range(cfg.num_epoch):
        global_step = train_one_epoch(
            model=model,
            header=header,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cfg=cfg,
            epoch=epoch,
            global_step=global_step,
            use_wandb=use_wandb,
        )

        print(f"[VAL] Starting validation after epoch {epoch + 1}", flush=True)
        val_auc, val_acc = validate(model, val_loader, device)
        print(
            f"[VAL] Finished epoch {epoch + 1}: "
            f"RFW AUC={val_auc:.6f}, RFW ACC={val_acc:.6f}",
            flush=True,
        )

        log_dict = {
            "val/rfw_auc": val_auc,
            "val/rfw_acc": val_acc,
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
        }

        if use_wandb:
            wandb.log(log_dict)
        else:
            print(log_dict, flush=True)

        if val_auc > best_val_auc:
            best_val_auc = val_auc

            save_checkpoint(
                path=best_path,
                model=model,
                header=header,
                cfg=cfg,
                best_val_auc=best_val_auc,
                num_classes=train_dataset.num_classes,
                epoch=epoch,
            )

            if use_wandb:
                wandb.run.summary["best_val_auc"] = best_val_auc
                wandb.run.summary["best_epoch"] = epoch

            print(f"[BEST] epoch={epoch} RFW AUC={best_val_auc:.6f}", flush=True)
            print(f"[BEST] saved to: {best_path}", flush=True)

        save_checkpoint(
            path=last_path,
            model=model,
            header=header,
            cfg=cfg,
            best_val_auc=best_val_auc,
            num_classes=train_dataset.num_classes,
            epoch=epoch,
        )

    if use_wandb:
        wandb.alert(
            title="FR pretraining done",
            text=f"Training finished. Best RFW AUC: {best_val_auc:.6f}",
            level=wandb.AlertLevel.INFO,
        )
        wandb.finish()

    print("Training done.", flush=True)
    print(f"Best RFW AUC: {best_val_auc:.6f}", flush=True)
    print(f"Best checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FR pretraining for DeepfakeBench detector backbones"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to FR pretraining config file",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    main(parser.parse_args())