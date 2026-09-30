import argparse
from pathlib import Path
import torch


# ============================================================
# Architecture inference
# ============================================================

def infer_arch_from_path(path):
    name = str(path).lower()

    if "efficientnetb4" in name or "efficientnet-b4" in name:
        return "efficientnetb4"

    if "resnet34" in name or "resnet-34" in name:
        return "resnet34"

    if "xception" in name:
        return "xception"

    if "swin" in name:
        return "swin"

    if "deit" in name:
        return "deit"

    if "clip" in name:
        return "clip"

    return "unknown"


# ============================================================
# Generic checkpoint extraction
# ============================================================

def extract_backbone_state(ckpt):
    if not isinstance(ckpt, dict):
        return ckpt, "raw_state_dict"

    if "detector_backbone" in ckpt:
        return ckpt["detector_backbone"], "detector_backbone"

    if "detector" in ckpt:
        return ckpt["detector"], "detector"

    # Common checkpoint formats
    if "state_dict" in ckpt:
        return ckpt["state_dict"], "state_dict"

    if "model" in ckpt:
        return ckpt["model"], "model"

    # Maybe checkpoint itself is already a state dict
    if len(ckpt) > 0 and all(torch.is_tensor(v) for v in ckpt.values()):
        return ckpt, "top_level_state_dict"

    raise ValueError(f"Could not find state dict. Top-level keys: {list(ckpt.keys())}")


def strip_prefixes(state_dict):
    prefixes_to_strip = [
        "module.",
        "detector_backbone.",
        "detector.",
        "backbone.",
        "model.",
        "net.",
        "encoder.",
        "feature_extractor.",
        "features.",
        "efficientnet.",
        "xception.",
        "resnet.",
    ]

    new_state = {}

    for k, v in state_dict.items():
        new_k = k

        changed = True
        while changed:
            changed = False
            for prefix in prefixes_to_strip:
                if new_k.startswith(prefix):
                    new_k = new_k[len(prefix):]
                    changed = True

        new_state[new_k] = v

    return new_state


# ============================================================
# ResNet34 fix
# ============================================================

def remap_resnet34_sequential_keys(state):
    """
    Convert Sequential-style ResNet keys to standard torchvision-style ResNet keys.

    From:
        0.weight
        1.weight
        4.0.conv1.weight
        5.0.conv1.weight
        6.0.conv1.weight
        7.0.conv1.weight

    To:
        conv1.weight
        bn1.weight
        layer1.0.conv1.weight
        layer2.0.conv1.weight
        layer3.0.conv1.weight
        layer4.0.conv1.weight
    """

    new_state = {}
    remapped = []

    for k, v in state.items():
        new_k = k

        if k.startswith("0."):
            new_k = "conv1." + k[len("0."):]
        elif k.startswith("1."):
            new_k = "bn1." + k[len("1."):]
        elif k.startswith("4."):
            new_k = "layer1." + k[len("4."):]
        elif k.startswith("5."):
            new_k = "layer2." + k[len("5."):]
        elif k.startswith("6."):
            new_k = "layer3." + k[len("6."):]
        elif k.startswith("7."):
            new_k = "layer4." + k[len("7."):]

        if new_k != k:
            remapped.append((k, new_k))

        new_state[new_k] = v

    return new_state, remapped


# ============================================================
# Xception / pointwise conv fix
# ============================================================

def fix_xception_pointwise_shapes(state):
    """
    Fix Xception SeparableConv2d pointwise weights accidentally saved as 6D.

    From:
        [out_channels, in_channels, 1, 1, 1, 1]

    To:
        [out_channels, in_channels, 1, 1]

    This is safe to apply globally because it only touches keys containing
    'pointwise.weight' with the exact bad 6D singleton pattern.
    """

    fixed = []

    for k, v in list(state.items()):
        if (
            "pointwise.weight" in k
            and torch.is_tensor(v)
            and v.ndim == 6
            and tuple(v.shape[-4:]) == (1, 1, 1, 1)
        ):
            state[k] = v.squeeze(-1).squeeze(-1)
            fixed.append(k)

    return state, fixed


def check_bad_pointwise_shapes(state):
    """
    Verify that no pointwise.weight tensors remain with non-4D shapes.
    """

    bad = []

    for k, v in state.items():
        if (
            "pointwise.weight" in k
            and torch.is_tensor(v)
            and v.ndim != 4
        ):
            bad.append((k, tuple(v.shape)))

    return bad


# ============================================================
# Remove incompatible detector/classifier keys
# ============================================================

def remove_arch_incompatible_keys(state, arch):
    """
    Remove keys that belong to detector-specific heads rather than the backbone
    expected by the downstream architecture loader.
    """

    removed = []

    for k in list(state.keys()):
        remove = False

        if arch == "efficientnetb4":
            # DeepfakeBench detector classifier, not part of efficientnet_pytorch EfficientNet.
            if k.startswith("last_layer."):
                remove = True

        if remove:
            removed.append(k)
            del state[k]

    return state, removed


# ============================================================
# Classifier compatibility keys
# ============================================================

def add_compatibility_keys(state, arch, num_classes=2):
    """
    Adds dummy classifier keys only when needed by strict downstream loaders.

    These classifier weights are not meaningful FR-pretraining weights.
    They only satisfy strict loaders. The downstream binary classifier/head
    should be trained normally.
    """

    added = []

    if arch == "efficientnetb4":
        # efficientnet_pytorch.from_pretrained() expects ImageNet-style FC
        # during loading, even if downstream later replaces/ignores it.
        feature_dim = 1792
        efficientnet_fc_classes = 1000

        if (
            "_fc.weight" not in state
            or tuple(state["_fc.weight"].shape) != (efficientnet_fc_classes, feature_dim)
        ):
            state["_fc.weight"] = torch.empty(efficientnet_fc_classes, feature_dim)
            torch.nn.init.normal_(state["_fc.weight"], mean=0.0, std=0.01)
            added.append("_fc.weight")

        if (
            "_fc.bias" not in state
            or tuple(state["_fc.bias"].shape) != (efficientnet_fc_classes,)
        ):
            state["_fc.bias"] = torch.zeros(efficientnet_fc_classes)
            added.append("_fc.bias")

    elif arch == "resnet34":
        feature_dim = 512

        if "fc.weight" not in state:
            state["fc.weight"] = torch.empty(num_classes, feature_dim)
            torch.nn.init.normal_(state["fc.weight"], mean=0.0, std=0.01)
            added.append("fc.weight")

        if "fc.bias" not in state:
            state["fc.bias"] = torch.zeros(num_classes)
            added.append("fc.bias")

    return state, added


# ============================================================
# Optional cleanup for FR-specific keys
# ============================================================

def remove_known_fr_head_keys(state):
    """
    Removes obvious FR-only keys if they slipped into the exported state.
    Usually they should not be present if detector_backbone was extracted,
    but this keeps the export cleaner.
    """

    remove_substrings = [
        "embedding_head",
        "fr_header",
        "arcface",
        "adaface",
        "elasticarcface",
    ]

    removed = []

    for k in list(state.keys()):
        lower_k = k.lower()
        if any(s in lower_k for s in remove_substrings):
            removed.append(k)
            del state[k]

    return state, removed


# ============================================================
# Export logic
# ============================================================

def export_one(src_path, output_name, overwrite=False, dry_run=False, num_classes=2):
    src_path = Path(src_path).resolve()
    dst_path = src_path.parent / output_name

    if dst_path == src_path:
        raise ValueError("Output path would overwrite source checkpoint.")

    if dst_path.exists() and not overwrite:
        print(f"[SKIP] Already exists: {dst_path}")
        return

    arch = infer_arch_from_path(src_path)

    print(f"\n[LOAD] {src_path}")
    print(f"[INFO] inferred_arch={arch}")
    print(f"[INFO] output={dst_path}")

    if dry_run:
        print(f"[DRY RUN] Would save to: {dst_path}")
        return

    ckpt = torch.load(src_path, map_location="cpu")

    state, source_key = extract_backbone_state(ckpt)
    state = strip_prefixes(state)
    state, removed_fr_keys = remove_known_fr_head_keys(state)

    remapped_resnet_keys = []

    # ResNet-specific key-name conversion
    if arch == "resnet34":
        state, remapped_resnet_keys = remap_resnet34_sequential_keys(state)

    # Apply globally, not only when arch == "xception".
    # This prevents missed fixes if arch inference fails.
    state, fixed_pointwise_keys = fix_xception_pointwise_shapes(state)

    # Hard verification: do not save if pointwise shapes are still invalid.
    bad_pointwise = check_bad_pointwise_shapes(state)
    if bad_pointwise:
        print("[ERROR] Remaining non-4D pointwise.weight tensors after fix:")
        for k, shape in bad_pointwise[:20]:
            print(f"  {k}: {shape}")
        raise RuntimeError("Export still contains non-4D pointwise.weight tensors.")

    # Remove detector-specific incompatible keys
    state, removed_incompatible_keys = remove_arch_incompatible_keys(state, arch)

    # Add dummy compatibility classifier keys where strict loaders require them
    state, added_keys = add_compatibility_keys(
        state=state,
        arch=arch,
        num_classes=num_classes,
    )

    torch.save(state, dst_path)

    print(f"[OK] source_key={source_key}")
    print(f"[OK] saved: {dst_path}")

    if removed_fr_keys:
        print(f"[INFO] removed FR-only keys: {len(removed_fr_keys)}")

    if removed_incompatible_keys:
        print(f"[INFO] removed incompatible keys: {removed_incompatible_keys}")

    if remapped_resnet_keys:
        print(f"[INFO] remapped ResNet keys: {len(remapped_resnet_keys)}")
        for old, new in remapped_resnet_keys[:8]:
            print(f"  {old} -> {new}")

    if fixed_pointwise_keys:
        print(f"[INFO] fixed pointwise weights: {len(fixed_pointwise_keys)}")
        for k in fixed_pointwise_keys[:8]:
            print(f"  {k}")

    if added_keys:
        print(f"[INFO] added compatibility keys: {added_keys}")

    print("[INFO] first exported keys:")
    for k in list(state.keys())[:8]:
        shape = tuple(state[k].shape) if torch.is_tensor(state[k]) else type(state[k])
        print(f"  {k}: {shape}")

    print("[INFO] last exported keys:")
    for k in list(state.keys())[-8:]:
        shape = tuple(state[k].shape) if torch.is_tensor(state[k]) else type(state[k])
        print(f"  {k}: {shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Export architecture-compatible backbone-only checkpoints from FR pretraining folders."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root folder containing checkpoint subfolders.",
    )

    parser.add_argument(
        "--checkpoint-name",
        type=str,
        default="best_fr_pretrained_detector_backbone.pth",
        help="Checkpoint filename to search for.",
    )

    parser.add_argument(
        "--output-name",
        type=str,
        default="backbone_only_compatible.pth",
        help="Output filename saved inside each checkpoint subfolder.",
    )

    parser.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of downstream classes for dummy classifier compatibility keys.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing exported files.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be written without creating files.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root folder not found: {root}")

    ckpts = sorted(root.rglob(args.checkpoint_name))

    print(f"[INFO] root: {root}")
    print(f"[INFO] found {len(ckpts)} checkpoint(s) named {args.checkpoint_name}")

    for ckpt_path in ckpts:
        ckpt_path = ckpt_path.resolve()

        # Extra safety: only process files inside subfolders, not directly under root.
        if ckpt_path.parent == root:
            print(f"[SKIP] Checkpoint is directly under root, not a subfolder: {ckpt_path}")
            continue

        try:
            export_one(
                src_path=ckpt_path,
                output_name=args.output_name,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                num_classes=args.num_classes,
            )
        except Exception as e:
            print(f"[ERROR] Failed on {ckpt_path}")
            print(f"        {type(e).__name__}: {e}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()