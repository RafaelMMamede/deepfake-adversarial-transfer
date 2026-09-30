import os
import json
import yaml
import random
import argparse
import datetime
from datetime import timedelta

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from optimizor.SAM import SAM
from optimizor.LinearLR import LinearDecayLR

from trainer.trainer import Trainer
from detectors import DETECTOR
from dataset.abstract_dataset import DeepfakeAbstractBaseDataset
from metrics.utils import parse_metric_for_print
from logger import create_logger
import torch.optim as optim

parser = argparse.ArgumentParser(description="Minimal DeepfakeBench train/val/test pipeline")
parser.add_argument(
    "--detector_path",
    type=str,
    required=True,
    help="Path to detector YAML file"
)
parser.add_argument(
    "--exp_config",
    type=str,
    required=True,
    help="Path to experiment JSON with train/val/test split definitions"
)
parser.add_argument("--no-save_ckpt", dest="save_ckpt", action="store_false", default=True)

parser.add_argument("--wandb", action="store_true", default=False)
parser.add_argument("--wandb_project", type=str, default="DeepfakeBench")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_run_name", type=str, default=None)
parser.add_argument("--wandb_tags", type=str, default=None, help="comma-separated tags")
parser.add_argument("--wandb_group", type=str, default=None)
parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
parser.add_argument("--wandb_dir", type=str, default=None)

args = parser.parse_args()


class WarmupThenScheduler:
    def __init__(self, optimizer, warmup_epochs, after_scheduler=None, start_factor=0.1):
        self.optimizer = optimizer
        self.warmup_epochs = int(warmup_epochs)
        self.after_scheduler = after_scheduler
        self.start_factor = float(start_factor)
        self.current_epoch = 0

        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Start at reduced LR
        if self.warmup_epochs > 0:
            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * self.start_factor

    def step(self):
        self.current_epoch += 1

        if self.warmup_epochs > 0 and self.current_epoch <= self.warmup_epochs:
            alpha = self.current_epoch / self.warmup_epochs
            factor = self.start_factor + alpha * (1.0 - self.start_factor)

            for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base_lr * factor

        else:
            if self.after_scheduler is not None:
                self.after_scheduler.step()

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_experiment_config(exp_config_path):
    exp_cfg = load_json(exp_config_path)

    if "train" not in exp_cfg:
        raise ValueError("Experiment config must contain 'train'.")
    if "val" not in exp_cfg:
        raise ValueError("Experiment config must contain 'val'.")

    for split_name in ["train", "val"]:
        split_cfg = exp_cfg[split_name]
        if not isinstance(split_cfg, dict):
            raise ValueError(f"Experiment split '{split_name}' must be a dict.")
        if "real" not in split_cfg or "fake" not in split_cfg:
            raise ValueError(f"Experiment split '{split_name}' must contain 'real' and 'fake'.")
        if not isinstance(split_cfg["real"], list) or not isinstance(split_cfg["fake"], list):
            raise ValueError(f"Experiment split '{split_name}' values must be lists.")

    if "test" in exp_cfg and not isinstance(exp_cfg["test"], dict):
        raise ValueError("Experiment config 'test' must be a dict of named splits.")

    return exp_cfg


def init_seed(config):
    if config.get("manualSeed", None) is None:
        config["manualSeed"] = random.randint(1, 10000)

    seed = int(config["manualSeed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.get("cuda", True) and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(config, mode, split_cfg):
    local_config = dict(config)
    local_config["active_split"] = split_cfg
    return DeepfakeAbstractBaseDataset(config=local_config, mode=mode)


def make_loader(dataset, batch_size, workers, shuffle=False, balanced_sampling=False):
    if balanced_sampling:
        labels = np.asarray(dataset.label_list, dtype=np.int64)
        num_classes = labels.max() + 1
        counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        class_weights = 1.0 / counts
        sample_weights = class_weights[labels]

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=int(workers),
            collate_fn=dataset.collate_fn,
            sampler=sampler,
            shuffle=False,
            prefetch_factor=2 if int(workers) > 0 else None,
        )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=int(workers),
        collate_fn=dataset.collate_fn,
        shuffle=shuffle,
        prefetch_factor=2 if int(workers) > 0 else None,
    )


def prepare_training_data(config):
    train_set = build_dataset(config, mode="train", split_cfg=config["train_split"])
    train_loader = make_loader(
        dataset=train_set,
        batch_size=config["train_batchSize"],
        workers=config["workers"],
        shuffle=not config.get("balanced_sampling", False),
        balanced_sampling=config.get("balanced_sampling", False),
    )
    return train_loader


def prepare_validation_data(config):
    val_set = build_dataset(config, mode="val", split_cfg=config["val_split"])
    val_loader = make_loader(
        dataset=val_set,
        batch_size=config["test_batchSize"],
        workers=config["workers"],
        shuffle=False,
        balanced_sampling=False,
    )
    return val_loader


def prepare_test_data(config):
    test_data_loaders = {}
    for split_name, split_cfg in config.get("test_splits", {}).items():
        test_set = build_dataset(config, mode="test", split_cfg=split_cfg)
        test_loader = make_loader(
            dataset=test_set,
            batch_size=config["test_batchSize"],
            workers=config["workers"],
            shuffle=False,
            balanced_sampling=False,
        )
        test_data_loaders[split_name] = test_loader
    return test_data_loaders


def get_param_groups(model, config):
    opt_name = config["optimizer"]["type"]
    opt_cfg = config["optimizer"][opt_name]

    # default: single group for models that don't define custom grouping
    if not hasattr(model, "get_param_groups"):
        return [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": opt_cfg["lr"],
            }
        ]

    return model.get_param_groups(config)


def choose_optimizer(model, config):
    opt_name = config["optimizer"]["type"]
    param_groups = get_param_groups(model, config)

    if opt_name == "sgd":
        return optim.SGD(
            params=param_groups,
            lr=config["optimizer"][opt_name]["lr"],  # fallback/default
            momentum=config["optimizer"][opt_name]["momentum"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
        )

    if opt_name == "adam":
        return optim.Adam(
            params=param_groups,
            lr=config["optimizer"][opt_name]["lr"],  # fallback/default
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=config["optimizer"][opt_name]["eps"],
            amsgrad=config["optimizer"][opt_name]["amsgrad"],
        )
    if opt_name == "adamw":
        return optim.AdamW(
            params=param_groups,
            lr=config["optimizer"][opt_name]["lr"],
            weight_decay=config["optimizer"][opt_name]["weight_decay"],
            betas=(
                config["optimizer"][opt_name]["beta1"],
                config["optimizer"][opt_name]["beta2"],
            ),
            eps=config["optimizer"][opt_name]["eps"],
            amsgrad=config["optimizer"][opt_name].get("amsgrad", False),
        )

    if opt_name == "sam":
        return SAM(
            param_groups,
            optim.SGD,
            lr=config["optimizer"][opt_name]["lr"],
            momentum=config["optimizer"][opt_name]["momentum"],
        )

    raise NotImplementedError(f"Optimizer {opt_name} is not implemented")


def choose_scheduler(config, optimizer):
    warmup_epochs = int(config.get("warmup_epochs", 0))
    warmup_start_factor = float(config.get("warmup_start_factor", 0.1))

    if config["lr_scheduler"] is None:
        base_scheduler = None

    elif config["lr_scheduler"] == "step":
        base_scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config["lr_step"],
            gamma=config["lr_gamma"],
        )

    elif config["lr_scheduler"] == "cosine":
        base_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["lr_T_max"],
            eta_min=config["lr_eta_min"],
        )

    elif config["lr_scheduler"] == "linear":
        base_scheduler = LinearDecayLR(
            optimizer,
            config["nEpochs"],
            int(config["nEpochs"] / 4),
        )

    else:
        raise NotImplementedError(f"Scheduler {config['lr_scheduler']} is not implemented")

    if warmup_epochs > 0:
        return WarmupThenScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            after_scheduler=base_scheduler,
            start_factor=warmup_start_factor,
        )

    return base_scheduler


def choose_metric(config):
    metric_scoring = config["metric_scoring"]
    if metric_scoring not in ["eer", "auc", "acc", "ap"]:
        raise NotImplementedError(f"Metric {metric_scoring} is not implemented")
    return metric_scoring


def init_wandb(config):
    import wandb

    tags = None
    if args.wandb_tags:
        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        group=args.wandb_group,
        tags=tags,
        config=config,
        mode=args.wandb_mode,
        dir=args.wandb_dir,
        settings=wandb.Settings(init_timeout=300, start_method="thread"),
    )
    return run


def main():
    detector_cfg = load_yaml(args.detector_path)
    train_cfg = load_yaml("./training/config/train_config.yaml")

    if "label_dict" in detector_cfg:
        train_cfg["label_dict"] = detector_cfg["label_dict"]

    config = {}
    config.update(detector_cfg)
    config.update(train_cfg)
    config["ddp"] = False

    exp_cfg = load_experiment_config(args.exp_config)
    config["exp_config_path"] = args.exp_config
    config["exp_name"] = exp_cfg.get("name", os.path.splitext(os.path.basename(args.exp_config))[0])
    config["train_split"] = exp_cfg["train"]
    config["val_split"] = exp_cfg["val"]
    config["test_splits"] = exp_cfg.get("test", {})
    config["save_ckpt"] = args.save_ckpt

    if config.get("dry_run", False):
        config["nEpochs"] = 0

    if "train_split" not in config or config["train_split"] is None:
        raise ValueError("Experiment config must define 'train'.")
    if "val_split" not in config or config["val_split"] is None:
        raise ValueError("Experiment config must define 'val'.")

    timenow = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    task_str = f"_{config['task_target']}" if config.get("task_target", None) is not None else ""
    logger_path = os.path.join(
        config["log_dir"],
        config["model_name"] + task_str + "_" + config["exp_name"] + "_" + timenow,
    )
    os.makedirs(logger_path, exist_ok=True)

    logger = create_logger(os.path.join(logger_path, "training.log"))
    logger.info(f"Save log to {logger_path}")

    wandb_run = None
    if args.wandb and args.wandb_mode != "disabled":
        wandb_run = init_wandb(config)

    logger.info("--------------- Configuration ---------------")
    params_string = "Parameters:\n"
    for key, value in config.items():
        params_string += f"{key}: {value}\n"
    logger.info(params_string)

    logger.info("--------------- Experiment Split Summary ---------------")
    logger.info(f"Experiment name: {config['exp_name']}")
    logger.info(f"Train split: {json.dumps(config['train_split'], indent=2)}")
    logger.info(f"Val split: {json.dumps(config['val_split'], indent=2)}")
    if config.get("test_splits", {}):
        logger.info(f"Test split names: {list(config['test_splits'].keys())}")

    init_seed(config)

    if config.get("cudnn", False):
        cudnn.benchmark = True

    train_data_loader = prepare_training_data(config)
    val_data_loader = prepare_validation_data(config)

    model_class = DETECTOR[config["model_name"]]
    model = model_class(config)

    optimizer = choose_optimizer(model, config)
    scheduler = choose_scheduler(config, optimizer)
    metric_scoring = choose_metric(config)

    for i, group in enumerate(optimizer.param_groups):
        nparams = sum(p.numel() for p in group["params"])
        print(f"group {i}: lr={group['lr']}, nparams={nparams}")

    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        metric_scoring=metric_scoring,
        wandb_run=wandb_run,
    )

    best_metric = None

    for epoch in range(config["start_epoch"], config["nEpochs"] + 1):
        if hasattr(trainer.model, "epoch"):
            trainer.model.epoch = epoch

        best_metric = trainer.train_epoch(
            epoch=epoch,
            train_data_loader=train_data_loader,
            val_data_loader=val_data_loader,
        )

        if scheduler is not None:
            scheduler.step()

        if best_metric is not None:
            logger.info(
                f"===> Epoch[{epoch}] end with validation {metric_scoring}: "
                f"{parse_metric_for_print(best_metric)}!"
            )

    if best_metric is not None:
        logger.info(
            "Stop Training on best Validation metric {}".format(
                parse_metric_for_print(best_metric)
            )
        )
    else:
        logger.info("Stop Training with no validation metric returned.")

    if config.get("test_splits", {}):
        logger.info("Preparing final test loaders...")
        test_data_loaders = prepare_test_data(config)

        best_ckpt_path = os.path.join(trainer.log_dir, "val", "val", "ckpt_best.pth")
        if os.path.isfile(best_ckpt_path):
            logger.info(f"Loading best validation checkpoint from: {best_ckpt_path}")
            trainer.load_ckpt(best_ckpt_path)

            logger.info("===> Final Test start!")
            trainer.test_epoch(
                epoch=config["nEpochs"],
                iteration=-1,
                test_data_loaders=test_data_loaders,
                step=config["nEpochs"] * max(1, len(train_data_loader)),
            )
            logger.info("===> Final Test finished!")
        else:
            logger.warning(f"Best validation checkpoint not found at {best_ckpt_path}. Skipping final test.")
    else:
        logger.info("No test_splits defined in experiment config. Skipping final test.")

    if "svdd" in config["model_name"]:
        model.update_R(config["nEpochs"])

    for writer in trainer.writers.values():
        writer.close()

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()