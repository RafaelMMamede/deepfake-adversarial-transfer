import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
project_root_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(project_root_dir)

import pickle
import datetime
import logging
import numpy as np
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.tensorboard import SummaryWriter
from metrics.base_metrics_class import Recorder
from torch.optim.swa_utils import AveragedModel, SWALR
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn import metrics
from metrics.utils import get_test_metrics
import gc
import pandas as pd
from pathlib import Path

FFpp_pool = ['FaceForensics++', 'FF-DF', 'FF-F2F', 'FF-FS', 'FF-NT']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer(object):
    def __init__(
        self,
        config,
        model,
        optimizer,
        scheduler,
        logger,
        metric_scoring='auc',
        time_now=datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
        swa_model=None,
        wandb_run=None,
    ):
        if config is None or model is None or optimizer is None or logger is None:
            raise ValueError("config, model, optimizer, logger must be implemented")

        self.config = config
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.swa_model = swa_model
        self.writers = {}
        self.logger = logger
        self.metric_scoring = metric_scoring
        self.wandb_run = wandb_run
        self._last_logged_step = -1

        self.best_metrics_all_time = defaultdict(
            lambda: defaultdict(
                lambda: float('-inf') if self.metric_scoring != 'eer' else float('inf')
            )
        )

        self.speed_up()

        self.timenow = time_now
        exp_name = self.config.get('exp_name', None)

        if exp_name is not None:
            if 'task_target' not in config:
                self.log_dir = os.path.join(
                    self.config['log_dir'],
                    self.config['model_name'] + '_' + exp_name + '_' + self.timenow
                )
            else:
                task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
                self.log_dir = os.path.join(
                    self.config['log_dir'],
                    self.config['model_name'] + task_str + '_' + exp_name + '_' + self.timenow
                )
        else:
            if 'task_target' not in config:
                self.log_dir = os.path.join(
                    self.config['log_dir'],
                    self.config['model_name'] + '_' + self.timenow
                )
            else:
                task_str = f"_{config['task_target']}" if config['task_target'] is not None else ""
                self.log_dir = os.path.join(
                    self.config['log_dir'],
                    self.config['model_name'] + task_str + '_' + self.timenow
                )

        os.makedirs(self.log_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # path helpers
    # -------------------------------------------------------------------------
    def _phase_dir(self, phase):
        phase_dir = Path(self.log_dir) / phase
        phase_dir.mkdir(parents=True, exist_ok=True)
        return phase_dir

    def _dataset_dir(self, phase, dataset_key):
        dataset_dir = self._phase_dir(phase) / dataset_key
        dataset_dir.mkdir(parents=True, exist_ok=True)
        return dataset_dir

    def _writer_dir(self, phase, dataset_key, metric_key):
        writer_dir = self._dataset_dir(phase, dataset_key) / metric_key / "metric_board"
        writer_dir.mkdir(parents=True, exist_ok=True)
        return writer_dir

    def _ckpt_path(self, phase, dataset_key):
        return self._dataset_dir(phase, dataset_key) / "ckpt_best.pth"

    def _metric_pickle_path(self, phase, dataset_key):
        return self._dataset_dir(phase, dataset_key) / "metric_dict_best.pickle"

    def _feature_path(self, phase, dataset_key):
        return self._dataset_dir(phase, dataset_key) / "feat_best.npy"

    def _data_dict_path(self, phase, dataset_key):
        return self._dataset_dir(phase, dataset_key) / f"data_dict_{phase}.pickle"

    def _results_table_csv_path(self, phase, step):
        return self._phase_dir(phase) / f"results_table_step_{step}.csv"

    def _results_table_pkl_path(self, phase, step):
        return self._phase_dir(phase) / f"results_table_step_{step}.pkl"

    # -------------------------------------------------------------------------
    # logging helpers
    # -------------------------------------------------------------------------
    def _get_log_step(self, step):
        safe_step = max(int(step), self._last_logged_step + 1)
        self._last_logged_step = safe_step
        return safe_step

    def _wandb_log(self, d: dict, step: int):
        if self.wandb_run is None:
            return
        if int(self.config.get("local_rank", 0)) != 0:
            return

        clean = {}
        for k, v in d.items():
            if v is None:
                continue
            if isinstance(v, (torch.Tensor, np.ndarray)):
                try:
                    if np.ndim(v) == 0:
                        v = float(v)
                    else:
                        continue
                except Exception:
                    continue
            clean[k] = v

        if clean:
            safe_step = self._get_log_step(step)
            self.wandb_run.log(clean, step=safe_step)

    def get_writer(self, phase, dataset_key, metric_key):
        writer_key = f"{phase}-{dataset_key}-{metric_key}"
        if writer_key not in self.writers:
            writer_path = str(self._writer_dir(phase, dataset_key, metric_key))
            self.writers[writer_key] = SummaryWriter(writer_path)
        return self.writers[writer_key]

    # -------------------------------------------------------------------------
    # setup / mode
    # -------------------------------------------------------------------------
    def speed_up(self):
        self.model.to(device)
        self.model.device = device
        if self.config['ddp'] is True:
            num_gpus = torch.cuda.device_count()
            print(f'avai gpus: {num_gpus}')
            self.model = DDP(
                self.model,
                device_ids=[self.config['local_rank']],
                find_unused_parameters=True,
                output_device=self.config['local_rank']
            )

    def setTrain(self):
        self.model.train()
        self.train = True

    def setEval(self):
        self.model.eval()
        self.train = False

    # -------------------------------------------------------------------------
    # save / load
    # -------------------------------------------------------------------------
    def load_ckpt(self, model_path):
        if os.path.isfile(model_path):
            saved = torch.load(model_path, map_location='cpu')
            suffix = model_path.split('.')[-1]
            if suffix == 'p':
                self.model.load_state_dict(saved.state_dict())
            else:
                self.model.load_state_dict(saved)
            self.logger.info('Model found in {}'.format(model_path))
        else:
            raise NotImplementedError("=> no model found at '{}'".format(model_path))

    def save_ckpt(self, phase, dataset_key, ckpt_info=None):
        save_path = str(self._ckpt_path(phase, dataset_key))

        if self.config['ddp'] is True:
            torch.save(self.model.state_dict(), save_path)
        else:
            if 'svdd' in self.config['model_name']:
                torch.save(
                    {
                        'R': self.model.R,
                        'c': self.model.c,
                        'state_dict': self.model.state_dict(),
                    },
                    save_path
                )
            else:
                torch.save(self.model.state_dict(), save_path)

        self.logger.info(f"Checkpoint saved to {save_path}, current ckpt is {ckpt_info}")

    def save_swa_ckpt(self):
        save_dir = self.log_dir
        os.makedirs(save_dir, exist_ok=True)
        ckpt_name = "swa.pth"
        save_path = os.path.join(save_dir, ckpt_name)
        torch.save(self.swa_model.state_dict(), save_path)
        self.logger.info(f"SWA Checkpoint saved to {save_path}")

    def save_feat(self, phase, fea, dataset_key):
        save_path = str(self._feature_path(phase, dataset_key))
        np.save(save_path, fea)
        self.logger.info(f"Feature saved to {save_path}")

    def save_data_dict(self, phase, data_dict, dataset_key):
        file_path = str(self._data_dict_path(phase, dataset_key))
        with open(file_path, 'wb') as file:
            pickle.dump(data_dict, file)
        self.logger.info(f"data_dict saved to {file_path}")

    def save_metrics(self, phase, metric_one_dataset, dataset_key):
        file_path = str(self._metric_pickle_path(phase, dataset_key))
        with open(file_path, 'wb') as file:
            pickle.dump(metric_one_dataset, file)
        self.logger.info(f"Metrics saved to {file_path}")

    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    def train_step(self, data_dict):
        if self.config['optimizer']['type'] == 'sam':
            for i in range(2):
                predictions = self.model(data_dict)
                losses = self.model.get_losses(data_dict, predictions)
                if i == 0:
                    pred_first = predictions
                    losses_first = losses
                self.optimizer.zero_grad()
                losses['overall'].backward()
                if i == 0:
                    self.optimizer.first_step(zero_grad=True)
                else:
                    self.optimizer.second_step(zero_grad=True)
            return losses_first, pred_first
        else:
            predictions = self.model(data_dict)
            if type(self.model) is DDP:
                losses = self.model.module.get_losses(data_dict, predictions)
            else:
                losses = self.model.get_losses(data_dict, predictions)
            self.optimizer.zero_grad()
            losses['overall'].backward()
            self.optimizer.step()
            return losses, predictions

    def train_epoch(
        self,
        epoch,
        train_data_loader,
        val_data_loader=None,
    ):
        self.logger.info("===> Epoch[{}] start!".format(epoch))

        if epoch >= 1:
            times_per_epoch = 2
        else:
            times_per_epoch = 1

        val_step = max(1, len(train_data_loader) // times_per_epoch)
        step_cnt = epoch * len(train_data_loader)
        val_best_metric = None

        train_recorder_loss = defaultdict(Recorder)
        train_recorder_metric = defaultdict(Recorder)

        log_every = int(self.config.get("train_log_iter", 300))
        postfix_every = int(self.config.get("tqdm_postfix_iter", 200))
        mininterval = float(self.config.get("tqdm_mininterval", 5.0))
        train_writer_key = self.config.get('exp_name', 'train')

        pbar = tqdm(
            train_data_loader,
            mininterval=mininterval,
        )

        last_overall_loss = None

        for iteration, data_dict in enumerate(pbar):
            self.setTrain()

            for key in data_dict.keys():
                if data_dict[key] is not None and key != 'name':
                    data_dict[key] = data_dict[key].cuda(non_blocking=True)

            losses, predictions = self.train_step(data_dict)

            try:
                last_overall_loss = float(losses["overall"].detach().item())
            except Exception:
                pass

            if postfix_every > 0 and (iteration % postfix_every == 0) and last_overall_loss is not None:
                pbar.set_postfix(loss=last_overall_loss)

            if type(self.model) is DDP:
                batch_metrics = self.model.module.get_train_metrics(data_dict, predictions)
            else:
                batch_metrics = self.model.get_train_metrics(data_dict, predictions)

            for name, value in batch_metrics.items():
                train_recorder_metric[name].update(value)
            for name, value in losses.items():
                train_recorder_loss[name].update(value)

            if (iteration % log_every == 0) and (self.config.get("local_rank", 0) == 0):
                loss_str = f"Iter: {step_cnt}    "
                metric_str = f"Iter: {step_cnt}    "

                wandb_payload = {"epoch": epoch, "iter": step_cnt}
                try:
                    wandb_payload["lr"] = float(self.optimizer.param_groups[0]["lr"])
                except Exception:
                    pass

                for k, v in train_recorder_loss.items():
                    v_avg = v.average()
                    if v_avg is None:
                        continue
                    loss_str += f"training-loss, {k}: {v_avg}    "
                    if self.config.get("use_tensorboard", True):
                        writer = self.get_writer('train', train_writer_key, k)
                        writer.add_scalar(f'train_loss/{k}', v_avg, global_step=step_cnt)
                    if self.config.get("use_wandb", True):
                        wandb_payload[f"train/loss/{k}"] = float(v_avg)

                for k, v in train_recorder_metric.items():
                    v_avg = v.average()
                    if v_avg is None:
                        continue
                    metric_str += f"training-metric, {k}: {v_avg}    "
                    if self.config.get("use_tensorboard", True):
                        writer = self.get_writer('train', train_writer_key, k)
                        writer.add_scalar(f'train_metric/{k}', v_avg, global_step=step_cnt)
                    if self.config.get("use_wandb", True):
                        wandb_payload[f"train/metric/{k}"] = float(v_avg)

                self.logger.info(loss_str)
                self.logger.info(metric_str)

                if self.config.get("use_wandb", True):
                    self._wandb_log(wandb_payload, step=step_cnt)

                for r in train_recorder_loss.values():
                    r.clear()
                for r in train_recorder_metric.values():
                    r.clear()

            if (step_cnt + 1) % val_step == 0:
                try:
                    del losses, predictions, batch_metrics
                except Exception:
                    pass
                gc.collect()
                torch.cuda.empty_cache()

                if val_data_loader is not None and (not self.config['ddp']):
                    self.logger.info("===> Validation start!")
                    val_best_metric = self.val_epoch(epoch, iteration, val_data_loader, step_cnt)
                elif val_data_loader is not None and (self.config['ddp'] and dist.get_rank() == 0):
                    self.logger.info("===> Validation start!")
                    val_best_metric = self.val_epoch(epoch, iteration, val_data_loader, step_cnt)

                gc.collect()
                torch.cuda.empty_cache()

            step_cnt += 1

        return val_best_metric

    # -------------------------------------------------------------------------
    # eval helpers
    # -------------------------------------------------------------------------
    def get_respect_acc(self, prob, label):
        pred = np.where(prob > 0.5, 1, 0)
        judge = (pred == label)
        real_idx = np.where(label == 0)[0]
        fake_idx = np.where(label == 1)[0]
        acc_real = np.count_nonzero(judge[real_idx]) / len(real_idx) if len(real_idx) > 0 else 0.0
        acc_fake = np.count_nonzero(judge[fake_idx]) / len(fake_idx) if len(fake_idx) > 0 else 0.0
        return acc_real, acc_fake

    def test_one_dataset(self, data_loader):
        test_recorder_loss = defaultdict(Recorder)
        prediction_lists = []
        label_lists = []

        for _, data_dict in tqdm(enumerate(data_loader), total=len(data_loader)):
            if 'label_spe' in data_dict:
                data_dict.pop('label_spe')

            data_dict['label'] = torch.where(data_dict['label'] != 0, 1, 0)

            for key in data_dict.keys():
                if data_dict[key] is not None:
                    data_dict[key] = data_dict[key].cuda()

            predictions = self.inference(data_dict)
            label_lists += list(data_dict['label'].cpu().detach().numpy())
            prediction_lists += list(predictions['prob'].cpu().detach().numpy())

            if type(self.model) is not AveragedModel:
                if type(self.model) is DDP:
                    losses = self.model.module.get_losses(data_dict, predictions)
                else:
                    losses = self.model.get_losses(data_dict, predictions)

                for name, value in losses.items():
                    test_recorder_loss[name].update(value)

        return test_recorder_loss, np.array(prediction_lists), np.array(label_lists)

    def _build_results_table(self, losses_all_datasets, metrics_all_datasets):
        rows = []

        for dataset_key in metrics_all_datasets.keys():
            row = {"dataset": dataset_key}

            loss_recorder = losses_all_datasets.get(dataset_key, None)
            if loss_recorder is not None:
                if hasattr(loss_recorder, "items"):
                    for loss_name, loss_value in loss_recorder.items():
                        try:
                            row[f"loss_{loss_name}"] = float(loss_value.average())
                        except Exception:
                            try:
                                row[f"loss_{loss_name}"] = float(loss_value)
                            except Exception:
                                row[f"loss_{loss_name}"] = None

            metric_dict = metrics_all_datasets.get(dataset_key, {})
            for metric_name, metric_value in metric_dict.items():
                if metric_name in ("pred", "label", "dataset_dict"):
                    continue
                try:
                    row[metric_name] = float(metric_value) if metric_value is not None else None
                except Exception:
                    row[metric_name] = metric_value

            if "pred" in metric_dict and "label" in metric_dict:
                try:
                    acc_real, acc_fake = self.get_respect_acc(metric_dict["pred"], metric_dict["label"])
                    row["acc_real"] = float(acc_real)
                    row["acc_fake"] = float(acc_fake)
                except Exception:
                    row["acc_real"] = None
                    row["acc_fake"] = None

            rows.append(row)

        df = pd.DataFrame(rows)

        preferred_cols = [
            "dataset",
            "loss_overall",
            "acc",
            "auc",
            "eer",
            "ap",
            "video_auc",
            "acc_real",
            "acc_fake",
        ]
        existing = [c for c in preferred_cols if c in df.columns]
        remaining = [c for c in df.columns if c not in existing]
        df = df[existing + remaining]

        return df

    # -------------------------------------------------------------------------
    # best/save + logging
    # -------------------------------------------------------------------------
    def save_best(self, epoch, iteration, step, losses_one_dataset_recorder, key, metric_one_dataset, phase='test'):
        best_metric = self.best_metrics_all_time[key].get(
            self.metric_scoring,
            float('-inf') if self.metric_scoring != 'eer' else float('inf')
        )

        current_score = metric_one_dataset.get(self.metric_scoring, None)

        improved = False
        if current_score is not None:
            improved = (
                current_score > best_metric
                if self.metric_scoring != 'eer'
                else current_score < best_metric
            )
        else:
            self.logger.warning(
                f"Metric '{self.metric_scoring}' is None for dataset={key}, phase={phase}; skipping best-model comparison."
            )

        if improved:
            self.best_metrics_all_time[key][self.metric_scoring] = current_score
            if key == 'avg' and 'dataset_dict' in metric_one_dataset:
                self.best_metrics_all_time[key]['dataset_dict'] = metric_one_dataset['dataset_dict']

            if self.config['save_ckpt'] and key not in FFpp_pool:
                self.save_ckpt(phase, key, f"{epoch}+{iteration}")
            self.save_metrics(phase, metric_one_dataset, key)

        if losses_one_dataset_recorder is not None:
            loss_str = f"dataset: {key}    step: {step}    "
            for k, v in losses_one_dataset_recorder.items():
                v_avg = v.average()
                if v_avg is None:
                    print(f'{k} is not calculated')
                    continue

                if self.config.get("use_tensorboard", True):
                    writer = self.get_writer(phase, key, k)
                    writer.add_scalar(f'{phase}_losses/{k}', v_avg, global_step=step)

                loss_str += f"{phase}-loss, {k}: {v_avg}    "
            self.logger.info(loss_str)

        metric_str = f"dataset: {key}    step: {step}    "
        for k, v in metric_one_dataset.items():
            if k in ('pred', 'label', 'dataset_dict'):
                continue

            metric_str += f"{phase}-metric, {k}: {v}    "

            if v is None:
                self.logger.warning(f"Skipping {phase}_metrics/{k} for dataset={key} because value is None")
                continue

            if self.config.get("use_tensorboard", True):
                writer = self.get_writer(phase, key, k)
                try:
                    writer.add_scalar(f'{phase}_metrics/{k}', float(v), global_step=step)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to log TensorBoard scalar {phase}_metrics/{k}={v} "
                        f"(type={type(v)}) for dataset={key}: {e}"
                    )

        acc_real, acc_fake = None, None
        if 'pred' in metric_one_dataset and 'label' in metric_one_dataset:
            acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
            metric_str += f'{phase}-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'

            if self.config.get("use_tensorboard", True):
                if acc_real is not None:
                    writer = self.get_writer(phase, key, 'acc_real')
                    writer.add_scalar(f'{phase}_metrics/acc_real', float(acc_real), global_step=step)

                if acc_fake is not None:
                    writer = self.get_writer(phase, key, 'acc_fake')
                    writer.add_scalar(f'{phase}_metrics/acc_fake', float(acc_fake), global_step=step)

        wandb_payload = {"epoch": epoch, "iter": step}
        for k, v in metric_one_dataset.items():
            if k in ("pred", "label", "dataset_dict"):
                continue
            if v is None:
                self.logger.warning(f"Skipping W&B log for {phase}/{key}/{k} because value is None")
                continue
            try:
                wandb_payload[f"{phase}/{key}/{k}"] = float(v)
            except Exception as e:
                self.logger.warning(
                    f"Skipping W&B log for {phase}/{key}/{k}={v} (type={type(v)}): {e}"
                )

        if acc_real is not None:
            wandb_payload[f"{phase}/{key}/acc_real"] = float(acc_real)
        if acc_fake is not None:
            wandb_payload[f"{phase}/{key}/acc_fake"] = float(acc_fake)

        try:
            best_val = self.best_metrics_all_time[key][self.metric_scoring]
            if best_val is not None:
                wandb_payload[f"best/{key}/{self.metric_scoring}"] = float(best_val)
        except Exception:
            pass

        if self.config.get("use_wandb", True):
            self._wandb_log(wandb_payload, step=step)

        self.logger.info(metric_str)

    # -------------------------------------------------------------------------
    # validation / test
    # -------------------------------------------------------------------------
    def val_epoch(self, epoch, iteration, val_data_loader, step):
        self.setEval()

        data_dict = val_data_loader.dataset.data_dict
        losses_one_dataset_recorder, predictions_nps, label_nps = self.test_one_dataset(val_data_loader)

        metric_one_dataset = get_test_metrics(
            y_pred=predictions_nps,
            y_true=label_nps,
            img_names=data_dict['image']
        )

        # ensure pred/label are available for acc_real/acc_fake and result tables
        if 'pred' not in metric_one_dataset:
            metric_one_dataset['pred'] = predictions_nps
        if 'label' not in metric_one_dataset:
            metric_one_dataset['label'] = label_nps

        if type(self.model) is AveragedModel:
            metric_str = "Iter Final for SWA:    "
            for k, v in metric_one_dataset.items():
                metric_str += f"validation-metric, {k}: {v}    "
            self.logger.info(metric_str)
        else:
            self.save_best(
                epoch,
                iteration,
                step,
                losses_one_dataset_recorder,
                'val',
                metric_one_dataset,
                phase='val'
            )

        self.logger.info('===> Validation Done!')
        return self.best_metrics_all_time

    def test_epoch(self, epoch, iteration, test_data_loaders, step):
        self.setEval()

        losses_all_datasets = {}
        metrics_all_datasets = {}

        avg_metric = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0, 'dataset_dict': {}}
        avg_metric_counts = {'acc': 0, 'auc': 0, 'eer': 0, 'ap': 0, 'video_auc': 0}

        keys = test_data_loaders.keys()
        for key in keys:
            data_dict = test_data_loaders[key].dataset.data_dict

            losses_one_dataset_recorder, predictions_nps, label_nps = self.test_one_dataset(test_data_loaders[key])
            losses_all_datasets[key] = losses_one_dataset_recorder

            metric_one_dataset = get_test_metrics(
                y_pred=predictions_nps,
                y_true=label_nps,
                img_names=data_dict['image']
            )

            # ensure pred/label are available for acc_real/acc_fake and result tables
            if 'pred' not in metric_one_dataset:
                metric_one_dataset['pred'] = predictions_nps
            if 'label' not in metric_one_dataset:
                metric_one_dataset['label'] = label_nps

            metrics_all_datasets[key] = metric_one_dataset

            for metric_name, value in metric_one_dataset.items():
                if metric_name in ('dataset_dict', 'pred', 'label'):
                    continue
                if metric_name in avg_metric and value is not None:
                    avg_metric[metric_name] += value
                    avg_metric_counts[metric_name] += 1

            avg_metric['dataset_dict'][key] = metric_one_dataset.get(self.metric_scoring, None)

            if type(self.model) is AveragedModel:
                metric_str = f"Iter Final for SWA:    "
                for k, v in metric_one_dataset.items():
                    metric_str += f"testing-metric, {k}: {v}    "
                self.logger.info(metric_str)
                continue

            # -----------------------------
            # log losses
            # -----------------------------
            if losses_one_dataset_recorder is not None:
                loss_str = f"dataset: {key}    step: {step}    "
                for k, v in losses_one_dataset_recorder.items():
                    v_avg = v.average()
                    if v_avg is None:
                        continue

                    if self.config.get("use_tensorboard", True):
                        writer = self.get_writer('test', key, k)
                        writer.add_scalar(f'test_losses/{k}', v_avg, global_step=step)

                    loss_str += f"test-loss, {k}: {v_avg}    "
                self.logger.info(loss_str)

            # -----------------------------
            # log metrics
            # -----------------------------
            metric_str = f"dataset: {key}    step: {step}    "
            for k, v in metric_one_dataset.items():
                if k in ('pred', 'label', 'dataset_dict'):
                    continue

                metric_str += f"test-metric, {k}: {v}    "

                if v is None:
                    self.logger.warning(f"Skipping test_metrics/{k} for dataset={key} because value is None")
                    continue

                if self.config.get("use_tensorboard", True):
                    writer = self.get_writer('test', key, k)
                    try:
                        writer.add_scalar(f'test_metrics/{k}', float(v), global_step=step)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to log TensorBoard scalar test_metrics/{k}={v} "
                            f"(type={type(v)}) for dataset={key}: {e}"
                        )

            acc_real, acc_fake = None, None
            if 'pred' in metric_one_dataset and 'label' in metric_one_dataset:
                acc_real, acc_fake = self.get_respect_acc(metric_one_dataset['pred'], metric_one_dataset['label'])
                metric_str += f'test-metric, acc_real:{acc_real}; acc_fake:{acc_fake}'

                if self.config.get("use_tensorboard", True):
                    if acc_real is not None:
                        writer = self.get_writer('test', key, 'acc_real')
                        writer.add_scalar('test_metrics/acc_real', float(acc_real), global_step=step)

                    if acc_fake is not None:
                        writer = self.get_writer('test', key, 'acc_fake')
                        writer.add_scalar('test_metrics/acc_fake', float(acc_fake), global_step=step)

            self.logger.info(metric_str)

            # save metrics only, no checkpoint
            self.save_metrics('test', metric_one_dataset, key)

            wandb_payload = {"epoch": epoch, "iter": step}
            for k, v in metric_one_dataset.items():
                if k in ("pred", "label", "dataset_dict"):
                    continue
                if v is None:
                    self.logger.warning(f"Skipping W&B log for test/{key}/{k} because value is None")
                    continue
                try:
                    wandb_payload[f"test/{key}/{k}"] = float(v)
                except Exception as e:
                    self.logger.warning(
                        f"Skipping W&B log for test/{key}/{k}={v} (type={type(v)}): {e}"
                    )

            if acc_real is not None:
                wandb_payload[f"test/{key}/acc_real"] = float(acc_real)
            if acc_fake is not None:
                wandb_payload[f"test/{key}/acc_fake"] = float(acc_fake)

            if self.config.get("use_wandb", True):
                self._wandb_log(wandb_payload, step=step)

        if len(keys) > 0 and self.config.get('save_avg', False):
            for metric_name in avg_metric_counts:
                if avg_metric_counts[metric_name] > 0:
                    avg_metric[metric_name] /= avg_metric_counts[metric_name]
                else:
                    avg_metric[metric_name] = None

            metrics_all_datasets['avg'] = avg_metric
            losses_all_datasets['avg'] = {}

            # log avg
            metric_str = f"dataset: avg    step: {step}    "
            for k, v in avg_metric.items():
                if k == 'dataset_dict':
                    continue

                metric_str += f"test-metric, {k}: {v}    "

                if v is None:
                    self.logger.warning(f"Skipping test_metrics/{k} for dataset=avg because value is None")
                    continue

                if self.config.get("use_tensorboard", True):
                    writer = self.get_writer('test', 'avg', k)
                    try:
                        writer.add_scalar(f'test_metrics/{k}', float(v), global_step=step)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to log TensorBoard scalar test_metrics/{k}={v} "
                            f"(type={type(v)}) for dataset=avg: {e}"
                        )

            self.logger.info(metric_str)
            self.save_metrics('test', avg_metric, 'avg')

            wandb_payload = {"epoch": epoch, "iter": step}
            for k, v in avg_metric.items():
                if k == "dataset_dict":
                    continue
                if v is None:
                    self.logger.warning(f"Skipping W&B log for test/avg/{k} because value is None")
                    continue
                try:
                    wandb_payload[f"test/avg/{k}"] = float(v)
                except Exception as e:
                    self.logger.warning(
                        f"Skipping W&B log for test/avg/{k}={v} (type={type(v)}): {e}"
                    )

            if self.config.get("use_wandb", True):
                self._wandb_log(wandb_payload, step=step)

        # save full test result table
        try:
            results_df = self._build_results_table(losses_all_datasets, metrics_all_datasets)

            csv_path = str(self._results_table_csv_path('test', step))
            pkl_path = str(self._results_table_pkl_path('test', step))

            results_df.to_csv(csv_path, index=False)
            results_df.to_pickle(pkl_path)

            self.logger.info(f"Test results table saved to {csv_path}")
            self.logger.info(f"Test results table saved to {pkl_path}")

            if self.wandb_run is not None and int(self.config.get("local_rank", 0)) == 0:
                try:
                    self.wandb_run.save(csv_path)
                except Exception as e:
                    self.logger.warning(f"Failed to save test results CSV to W&B: {e}")
        except Exception as e:
            self.logger.warning(f"Failed to build/save test results table: {e}")

        self.logger.info('===> Test Done!')
        return self.best_metrics_all_time

    # -------------------------------------------------------------------------
    # inference
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def inference(self, data_dict):
        predictions = self.model(data_dict, inference=True)
        return predictions