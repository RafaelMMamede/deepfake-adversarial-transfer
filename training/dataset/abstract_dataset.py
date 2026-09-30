"""
DeepfakeBench dataset base with support for experiment-defined binary splits.

Key additions:
- Supports config['active_split'] = {"real": [...], "fake": [...]}
- Supports mode in {'train', 'val', 'test'}
- For active_split:
    - real datasets are assigned label 0
    - fake datasets are assigned label 1
- For train/val, each atomic JSON can contain separate "train" and "val" sections
- Keeps backward compatibility with old train_dataset / test_dataset usage
- Still supports optional family supervision (multihead training)

Expected experiment usage:
    config['active_split'] = {
        "real": ["DF40_TRAIN_real_ff"],
        "fake": ["DF40_TRAIN_FR_ff"]
    }

Then:
    mode='train' -> read each source JSON's "train" section
    mode='val'   -> read each source JSON's "val" section
    mode='test'  -> read each source JSON's "test" section
"""

import os
import json
import random
import re
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T

import albumentations as A
from .albu import IsotropicResize


def _frame_sort_key(p: str) -> int:
    base = os.path.basename(p.replace("\\", "/"))
    stem = os.path.splitext(base)[0]
    if stem.isdigit():
        return int(stem)
    m = re.search(r"\d+", stem)
    if m:
        return int(m.group(0))
    return 10**18


class DeepfakeAbstractBaseDataset(data.Dataset):
    """
    Minimal dataset for experiment-defined binary splits.

    Required config keys:
      - active_split: {"real": [...], "fake": [...]}
      - dataset_json_folder
      - dataset_root_rgb
      - frame_num: {"train": ..., "val": ..., "test": ...}
      - resolution
      - mean, std
      - with_mask
      - with_landmark
      - use_data_augmentation
      - data_aug
    """

    def __init__(self, config=None, mode='train'):
        self.config = config
        self.mode = mode

        if mode not in ['train', 'val', 'test']:
            raise NotImplementedError("Only train, val and test modes are supported.")

        if 'active_split' not in config or config['active_split'] is None:
            raise ValueError("config['active_split'] must be provided.")

        self.frame_num = config['frame_num'][mode]

        self.image_list = []
        self.label_list = []

        image_list, label_list = self._build_samples()

        assert len(image_list) > 0 and len(label_list) > 0, f"Collected nothing for mode={mode}"

        self.image_list = image_list
        self.label_list = label_list

        self.data_dict = {
            'image': self.image_list,
            'label': self.label_list
        }

        self.transform = self.init_data_aug_method()

    def _build_samples(self):
        image_list, label_list = [], []

        split_cfg = self.config['active_split']
        if not isinstance(split_cfg, dict):
            raise ValueError("config['active_split'] must be a dict with keys 'real' and 'fake'.")

        real_datasets = split_cfg.get('real', [])
        fake_datasets = split_cfg.get('fake', [])

        if not isinstance(real_datasets, list) or not isinstance(fake_datasets, list):
            raise ValueError("config['active_split']['real'] and ['fake'] must be lists.")

        for ds_name in real_datasets:
            tmp_image, tmp_label, _ = self.collect_img_and_label_for_one_dataset(
                dataset_name=ds_name,
                forced_label_id=0
            )
            image_list.extend(tmp_image)
            label_list.extend(tmp_label)

        for ds_name in fake_datasets:
            tmp_image, tmp_label, _ = self.collect_img_and_label_for_one_dataset(
                dataset_name=ds_name,
                forced_label_id=1
            )
            image_list.extend(tmp_image)
            label_list.extend(tmp_label)

        return image_list, label_list

    def init_data_aug_method(self):
        trans = A.Compose([
            A.HorizontalFlip(p=self.config['data_aug']['flip_prob']),
            A.Rotate(limit=self.config['data_aug']['rotate_limit'], p=self.config['data_aug']['rotate_prob']),
            A.GaussianBlur(
                blur_limit=self.config['data_aug']['blur_limit'],
                p=self.config['data_aug']['blur_prob']
            ),
            A.OneOf([
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_CUBIC
                ),
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_AREA,
                    interpolation_up=cv2.INTER_LINEAR
                ),
                IsotropicResize(
                    max_side=self.config['resolution'],
                    interpolation_down=cv2.INTER_LINEAR,
                    interpolation_up=cv2.INTER_LINEAR
                ),
            ], p=0 if self.config['with_landmark'] else 1),
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=self.config['data_aug']['brightness_limit'],
                    contrast_limit=self.config['data_aug']['contrast_limit']
                ),
                A.FancyPCA(),
                A.HueSaturationValue()
            ], p=0.5),
            A.ImageCompression(
                quality_lower=self.config['data_aug']['quality_lower'],
                quality_upper=self.config['data_aug']['quality_upper'],
                p=0.5
            )
        ], keypoint_params=A.KeypointParams(format='xy') if self.config['with_landmark'] else None)
        return trans

    def _resolve_json_path(self, dataset_name: str) -> str:
        dataset_json_folder = self.config['dataset_json_folder']
        if not os.path.exists(dataset_json_folder):
            dataset_json_folder = dataset_json_folder.replace(
                '/Youtu_Pangu_Security_Public',
                '/Youtu_Pangu_Security/public'
            )
        return os.path.join(dataset_json_folder, dataset_name + '.json')

    def _load_dataset_json(self, dataset_name: str):
        json_path = self._resolve_json_path(dataset_name)
        try:
            with open(json_path, 'r') as f:
                dataset_info = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load dataset json: {json_path}. Error: {e}")
        return dataset_info, json_path

    def _normalize_frame_paths(self, frame_paths):
        root = os.path.normpath(self.config.get("dataset_root_rgb", "./datasets"))
        fixed = []
        for p in frame_paths:
            p2 = p.replace("\\", "/")
            if p2.startswith("../datasets/"):
                p2 = p2.replace("../datasets", root, 1)
            elif p2.startswith("./datasets/"):
                p2 = p2.replace("./datasets", root, 1)
            elif p2.startswith("datasets/"):
                p2 = os.path.join(root, p2[len("datasets/"):])
            fixed.append(os.path.normpath(p2))
        return fixed

    def _extract_entries_from_json(self, dataset_name: str, dataset_info: dict, json_path: str):
        """
        Supports:

        A)
        {
        "train": {...},
        "val": {...},
        "test": {...}
        }

        B)
        {
        "<dataset_name>": {
            "train": {...},
            "val": {...},
            "test": {...}
        }
        }

        C)
        {
        "<dataset_name>": {
            "<split_key>": {
            "train": {...},
            "val": {...},
            "test": {...}
            }
        }
        }
        """
        # Case A: direct top-level mode
        if self.mode in dataset_info:
            sub_dataset_info = dataset_info[self.mode]
            if not isinstance(sub_dataset_info, dict):
                raise ValueError(
                    f"Expected dict for mode '{self.mode}' in {json_path}, got {type(sub_dataset_info)}"
                )
            return list(sub_dataset_info.items())

        # Case B/C: wrapped by dataset name
        if dataset_name in dataset_info:
            candidate = dataset_info[dataset_name]
        else:
            raise KeyError(
                f"Could not resolve dataset '{dataset_name}' in {json_path}. "
                f"Available top-level keys: {list(dataset_info.keys())[:30]}"
            )

        # Case B: candidate has direct mode keys
        if self.mode in candidate:
            sub_dataset_info = candidate[self.mode]
            if not isinstance(sub_dataset_info, dict):
                raise ValueError(
                    f"Expected dict for mode '{self.mode}' in dataset '{dataset_name}' from {json_path}"
                )
            return list(sub_dataset_info.items())

        # Case C: candidate has an extra split-key layer
        entries = []
        for split_key, split_block in candidate.items():
            if not isinstance(split_block, dict):
                continue
            if self.mode not in split_block:
                continue

            sub_dataset_info = split_block[self.mode]
            if not isinstance(sub_dataset_info, dict):
                raise ValueError(
                    f"Expected dict for mode '{self.mode}' under split '{split_key}' "
                    f"in dataset '{dataset_name}' from {json_path}"
                )

            entries.extend(list(sub_dataset_info.items()))

        if len(entries) == 0:
            raise KeyError(
                f"Mode '{self.mode}' not found for dataset '{dataset_name}' in {json_path}. "
                f"Top-level keys: {list(dataset_info.keys())[:20]}"
            )

        return entries

    def collect_img_and_label_for_one_dataset(self, dataset_name: str, forced_label_id: int):
        frame_path_list = []
        label_list = []
        video_name_list = []

        dataset_info, json_path = self._load_dataset_json(dataset_name)
        entries = self._extract_entries_from_json(dataset_name, dataset_info, json_path)

        for video_name, video_info in entries:
            label_id = int(forced_label_id)
            label_str = 'fake' if label_id == 1 else 'real'

            frame_paths = video_info.get('frames', None)
            if frame_paths is None:
                raise KeyError(
                    f"'frames' key missing for video '{video_name}' in dataset '{dataset_name}'"
                )

            unique_video_name = f"{label_str}_{video_name}"

            frame_paths = self._normalize_frame_paths(frame_paths)
            frame_paths = sorted(frame_paths, key=_frame_sort_key)

            total_frames = len(frame_paths)
            if total_frames == 0:
                continue

            if self.frame_num < total_frames:
                step = max(1, total_frames // self.frame_num)
                frame_paths = [frame_paths[i] for i in range(0, total_frames, step)][:self.frame_num]
                total_frames = len(frame_paths)

            label_list.extend([label_id] * total_frames)
            frame_path_list.extend(frame_paths)
            video_name_list.extend([unique_video_name] * total_frames)

        if len(frame_path_list) == 0:
            return [], [], []

        pack = list(zip(frame_path_list, label_list, video_name_list))
        random.shuffle(pack)
        frame_path_list, label_list, video_name_list = zip(*pack)
        return list(frame_path_list), list(label_list), list(video_name_list)

    def load_rgb(self, file_path: str) -> Image.Image:
        size = int(self.config['resolution'])
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist")

        img = cv2.imread(file_path)
        if img is None:
            raise ValueError(f"cv2.imread returned None: {file_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(img.astype(np.uint8))

    def load_mask(self, file_path: str) -> np.ndarray:
        size = int(self.config['resolution'])
        if file_path is None or not os.path.exists(file_path):
            return np.zeros((size, size, 1), dtype=np.float32)

        mask = cv2.imread(file_path, 0)
        if mask is None:
            mask = np.zeros((size, size), dtype=np.uint8)

        mask = cv2.resize(mask, (size, size)) / 255.0
        mask = np.expand_dims(mask, axis=2)
        return mask.astype(np.float32)

    def load_landmark(self, file_path: str) -> np.ndarray:
        if file_path is None or not os.path.exists(file_path):
            return np.zeros((81, 2), dtype=np.float32)
        lm = np.load(file_path)
        return lm.astype(np.float32)

    def to_tensor(self, img):
        return T.ToTensor()(img)

    def normalize(self, img):
        mean = self.config['mean']
        std = self.config['std']
        return T.Normalize(mean=mean, std=std)(img)

    def data_aug(self, img, landmark=None, mask=None):
        kwargs = {'image': img}
        if landmark is not None:
            kwargs['keypoints'] = landmark
        if mask is not None:
            kwargs['mask'] = mask

        transformed = self.transform(**kwargs)

        aug_img = transformed['image']
        aug_lm = transformed.get('keypoints', None)
        aug_mask = transformed.get('mask', None)

        if aug_lm is not None:
            aug_lm = np.array(aug_lm, dtype=np.float32)

        return aug_img, aug_lm, aug_mask

    def __getitem__(self, index, no_norm=False):
        image_path = self.data_dict['image'][index]
        label = int(self.data_dict['label'][index])

        mask_path = image_path.replace('frames', 'masks')
        landmark_path = image_path.replace('frames', 'landmarks').replace('.png', '.npy')

        image = np.array(self.load_rgb(image_path))
        mask = self.load_mask(mask_path) if self.config['with_mask'] else None
        landmarks = self.load_landmark(landmark_path) if self.config['with_landmark'] else None

        if self.mode == 'train' and self.config['use_data_augmentation']:
            image_trans, landmarks_trans, mask_trans = self.data_aug(image, landmarks, mask)
        else:
            image_trans, landmarks_trans, mask_trans = deepcopy(image), deepcopy(landmarks), deepcopy(mask)

        if not no_norm:
            image_trans = self.normalize(self.to_tensor(image_trans))
            if self.config['with_landmark'] and landmarks_trans is not None:
                landmarks_trans = torch.from_numpy(landmarks_trans)
            if self.config['with_mask'] and mask_trans is not None:
                mask_trans = torch.from_numpy(mask_trans)

        return image_trans, label, landmarks_trans, mask_trans

    @staticmethod
    def collate_fn(batch):
        images, labels, landmarks, masks = zip(*batch)

        images = torch.stack(images, dim=0)
        labels = torch.LongTensor(labels)

        if landmarks is not None and not any(lm is None for lm in landmarks):
            landmarks = torch.stack(landmarks, dim=0)
        else:
            landmarks = None

        if masks is not None and not any(m is None for m in masks):
            masks = torch.stack(masks, dim=0)
        else:
            masks = None

        return {
            'image': images,
            'label': labels,
            'landmark': landmarks,
            'mask': masks
        }

    def __len__(self):
        assert len(self.image_list) == len(self.label_list)
        return len(self.image_list)