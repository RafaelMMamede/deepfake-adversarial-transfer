import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# Face-recognition pretraining dataset
# ============================================================

class RaceFaceFRPretrainDataset(Dataset):
    """
    Dataset for BUPT-BalancedFace / BUPT-GlobalFace style FR pretraining.

    Expected structure:
        dataset_dir/
            African/
                identity_1/
                    img1.jpg
                    img2.jpg
            Asian/
            Caucasian/
            Indian/

    Returns:
        image: Tensor [3, image_size, image_size]
        label: LongTensor identity class index
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(
        self,
        dataset_dir,
        ethnicity="All",
        image_size=224,
        is_train=True,
        transform=None,
        return_path=False,
        min_images_per_identity=1,
        cache_dir="./pretraining/cache",
        use_cache=True,
        rebuild_cache=False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.ethnicity = ethnicity
        self.image_size = image_size
        self.is_train = is_train
        self.return_path = return_path
        self.min_images_per_identity = min_images_per_identity
        self.cache_dir = Path(cache_dir)
        self.use_cache = use_cache
        self.rebuild_cache = rebuild_cache

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        if transform is None:
            self.transform = self._build_default_transform()
        else:
            self.transform = transform

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.image_paths, self.labels, self.label_to_index = self._load_dataset_cached()
        self.num_classes = len(self.label_to_index)

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.dataset_dir}. "
                f"Check directory structure and ethnicity={ethnicity}."
            )

        print("=" * 80, flush=True)
        print("[RaceFaceFRPretrainDataset]", flush=True)
        print(f"Dataset dir: {self.dataset_dir}", flush=True)
        print(f"Ethnicity filter: {self.ethnicity}", flush=True)
        print(f"Images: {len(self.image_paths)}", flush=True)
        print(f"Identities/classes: {self.num_classes}", flush=True)
        print(f"Image size: {self.image_size}", flush=True)
        print(f"Train mode: {self.is_train}", flush=True)
        print("=" * 80, flush=True)

    def _cache_path(self):
        dataset_tag = self.dataset_dir.name
        ethnicity_tag = str(self.ethnicity).replace("/", "_")
        return self.cache_dir / (
            f"{dataset_tag}_{ethnicity_tag}"
            f"_min{self.min_images_per_identity}_index.csv"
        )

    def _load_dataset_cached(self):
        cache_path = self._cache_path()

        if self.use_cache and cache_path.exists() and not self.rebuild_cache:
            print(f"[INFO] Loading dataset index cache: {cache_path}", flush=True)

            import pandas as pd

            df = pd.read_csv(cache_path)

            image_paths = df["path"].astype(str).tolist()
            labels = df["label"].astype(int).tolist()

            label_to_index = (
                df[["identity", "label"]]
                .drop_duplicates()
                .set_index("identity")["label"]
                .astype(int)
                .to_dict()
            )

            print(
                f"[INFO] Loaded cache with {len(image_paths)} images "
                f"and {len(label_to_index)} identities.",
                flush=True,
            )

            return image_paths, labels, label_to_index

        print("[INFO] No cache found, scanning dataset folders...", flush=True)
        image_paths, labels, label_to_index = self._load_dataset()

        if self.use_cache:
            print(f"[INFO] Saving dataset index cache: {cache_path}", flush=True)
            self._save_cache(cache_path, image_paths, labels, label_to_index)

        return image_paths, labels, label_to_index

    def _save_cache(self, cache_path, image_paths, labels, label_to_index):
        import pandas as pd

        index_to_label = {v: k for k, v in label_to_index.items()}

        rows = []
        for path, label in zip(image_paths, labels):
            rows.append({
                "path": str(path),
                "label": int(label),
                "identity": index_to_label[int(label)],
            })

        df = pd.DataFrame(rows)
        df.to_csv(cache_path, index=False)

    def _build_default_transform(self):
        if self.is_train:
            return transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=0.15,
                        contrast=0.15,
                        saturation=0.10,
                        hue=0.02,
                    )
                ], p=0.3),
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=5,
                        translate=(0.03, 0.03),
                        scale=(0.97, 1.03),
                    )
                ], p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                ),
            ])

        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

    def _get_ethnicity_dirs(self):
        if str(self.ethnicity).lower() == "all":
            ethnicity_dirs = [
                p for p in sorted(self.dataset_dir.iterdir())
                if p.is_dir()
            ]
        else:
            ethnicity_dir = self.dataset_dir / self.ethnicity

            if not ethnicity_dir.exists():
                available = [
                    p.name for p in sorted(self.dataset_dir.iterdir())
                    if p.is_dir()
                ]
                raise FileNotFoundError(
                    f"Ethnicity directory not found: {ethnicity_dir}\n"
                    f"Available folders: {available}"
                )

            ethnicity_dirs = [ethnicity_dir]

        return ethnicity_dirs

    def _load_dataset(self):
        image_paths = []
        labels = []

        label_to_index = {}
        label_counter = 0

        ethnicity_dirs = self._get_ethnicity_dirs()

        for ethnicity_dir in ethnicity_dirs:
            identity_dirs = [
                p for p in sorted(ethnicity_dir.iterdir())
                if p.is_dir()
            ]

            for identity_dir in identity_dirs:
                imgs = [
                    p for p in sorted(identity_dir.iterdir())
                    if p.is_file() and p.suffix.lower() in self.IMG_EXTS
                ]

                if len(imgs) < self.min_images_per_identity:
                    continue

                identity_key = f"{ethnicity_dir.name}/{identity_dir.name}"

                label_to_index[identity_key] = label_counter
                label = label_counter
                label_counter += 1

                for img_path in imgs:
                    image_paths.append(str(img_path))
                    labels.append(label)

        return image_paths, labels, label_to_index

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return image, label, image_path

        return image, label


# ============================================================
# RFW verification validation dataset
# ============================================================

class RFWVerificationDataset(Dataset):
    """
    Dataset class for RFW verification.

    Expected structure:
        dataset_dir/
            data/
                African/
                Asian/
                Caucasian/
                Indian/
            txts/
                African/
                    African_pairs.txt
                Asian/
                    Asian_pairs.txt
                ...

    Returns:
        img1, img2, label
    """

    def __init__(
        self,
        dataset_dir,
        image_size=224,
        transform=None,
        ethnicity=None,
        return_path=False,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.image_size = image_size
        self.ethnicity = ethnicity
        self.return_path = return_path

        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"RFW directory not found: {self.dataset_dir}")

        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5],
            ),
        ])

        self.pairs_files = self._detect_pairs_files()
        self.pairs = self._read_all_pairs()

        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No valid RFW pairs found in {self.dataset_dir}. "
                f"Check paths and ethnicity={ethnicity}."
            )

        print("=" * 80)
        print("[RFWVerificationDataset]")
        print(f"Dataset dir: {self.dataset_dir}")
        print(f"Ethnicity filter: {self.ethnicity}")
        print(f"Pairs: {len(self.pairs)}")
        print(f"Image size: {self.image_size}")
        print("=" * 80)

    def _detect_pairs_files(self):
        txts_dir = self.dataset_dir / "txts"

        if not txts_dir.exists():
            raise FileNotFoundError(f"Txts directory not found: {txts_dir}")

        pairs_files = []

        for ethnicity_folder in sorted(os.listdir(txts_dir)):
            if self.ethnicity is not None and ethnicity_folder != self.ethnicity:
                continue

            ethnicity_path = txts_dir / ethnicity_folder

            if not ethnicity_path.is_dir():
                continue

            pairs_file = ethnicity_path / f"{ethnicity_folder}_pairs.txt"

            if pairs_file.exists():
                pairs_files.append((ethnicity_folder, pairs_file))
            else:
                print(f"Warning: pairs file not found for ethnicity: {ethnicity_folder}")

        if not pairs_files:
            raise FileNotFoundError(
                f"No [Race]_pairs.txt files found under {txts_dir}"
            )

        return pairs_files

    def _read_all_pairs(self):
        all_pairs = []

        for ethnicity, pairs_file in self.pairs_files:
            data_dir = self.dataset_dir / "data" / ethnicity

            if not data_dir.exists():
                print(f"Warning: data directory not found: {data_dir}")
                continue

            with open(pairs_file, "r") as f:
                for line in f:
                    parts = line.strip().split("\t")

                    if len(parts) == 3:
                        # Same identity
                        folder_name, img1, img2 = parts
                        label = 1

                        img1_path = data_dir / folder_name / f"{folder_name}_000{img1}.jpg"
                        img2_path = data_dir / folder_name / f"{folder_name}_000{img2}.jpg"

                    elif len(parts) == 4:
                        # Different identities
                        folder_name1, img1, folder_name2, img2 = parts
                        label = 0

                        img1_path = data_dir / folder_name1 / f"{folder_name1}_000{img1}.jpg"
                        img2_path = data_dir / folder_name2 / f"{folder_name2}_000{img2}.jpg"

                    else:
                        print(f"Skipping malformed line: {line.strip()}")
                        continue

                    if not img1_path.exists() or not img2_path.exists():
                        print(f"Image not found: {img1_path} or {img2_path}")
                        continue

                    all_pairs.append((str(img1_path), str(img2_path), label, ethnicity))

        return all_pairs

    def _load_image(self, img_path):
        image = cv2.imread(img_path)

        if image is None:
            raise FileNotFoundError(f"cv2.imread failed for: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        img1_path, img2_path, label, ethnicity = self.pairs[index]

        img1 = self._load_image(img1_path)
        img2 = self._load_image(img2_path)

        label = torch.tensor(label, dtype=torch.long)

        if self.return_path:
            return img1, img2, label, img1_path, img2_path, ethnicity

        return img1, img2, label