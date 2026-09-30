import os

import math


class config:
    # ============================================================
    # Dataset paths
    # ============================================================
    rec = os.environ.get("BALANCEDFACE_ROOT", "./data/BalancedFace/race_per_7000_aligned")
    ethnicity = "All"

    rfw_dir = os.environ.get("RFW_ROOT", "./data/RFW/test_aligned")
    rfw_ethnicity = None

    image_size = 224
    rfw_image_size = 224
    min_images_per_identity = 1

    cache_dir = "./pretraining/cache"
    use_cache = True
    rebuild_cache = False


    # ============================================================
    # Detector-class mode
    # ============================================================
    network = "deit"
    use_detector_class = True

    detector_name = "deit_detector_custom"
    detector_modules = []

    # DeiT-small feature dimension
    feature_dim = 384

    detector_loss_func = "cross_entropy"

    detector_config = {
        "backbone_model": "deit_small_patch16_224",

        # Keep False to avoid ImageNet pretraining confound
        "pretrained": False,
        "checkpoint_path": None,

        "img_size": 224,
        "in_chans": 3,

        # Mild transformer regularization
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.1,

        # Mostly unused during FR pretraining if you only use .features(),
        # but needed for detector construction.
        "classifier_hidden_dim": 256,
        "classifier_dropout": 0.2,

        "loss_func": "cross_entropy",
    }

    # ============================================================
    # FR head
    # ============================================================
    embedding_size = 512
    embedding_dropout = 0.0

    loss = "ElasticArcFacePlus"

    # Softer than standard ArcFace settings; safer for random-init DeiT
    s = 32.0
    m = 0.3
    std = 0.05

    # Only used if loss = "AdaFace"
    h = 0.333
    t_alpha = 0.01

    # ============================================================
    # Training
    # ============================================================
    batch_size = 256
    val_batch_size = 2048

    optimizer = "adamw"

    # actual_lr = lr / 512 * batch_size
    #
    # With batch_size = 256:
    # actual_lr = lr / 2
    #
    # Therefore lr = 2e-4 gives actual optimizer LR = 1e-4.
    lr = 2e-4

    adamw_betas = (0.9, 0.999)
    adamw_eps = 1e-8

    weight_decay = 0.05

    # Unused by AdamW, but harmless for compatibility
    momentum = 0.9

    num_epoch = 30

    # Counts inside num_epoch, not added on top.
    warmup_epochs = 3


    num_workers = 12
    prefetch_factor = 2
    persistent_workers = True
    pin_memory = True

    grad_clip = 1.0

    use_amp = False


    log_interval = 100

    @staticmethod
    def lr_func(epoch):
        """
        Cosine LR multiplier.

        Warmup is applied separately in the training script as:
            final_factor = warmup_factor * lr_func(epoch_float)

        So this function should only define the post-warmup schedule shape.
        """
        min_lr_ratio = 0.01

        progress = min(epoch / config.num_epoch, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    # ============================================================
    # Logging / outputs
    # ============================================================
    use_wandb = False
    wandb_project = "fr_pretrain_deepfake_backbones"

    wandb_job_name = (
        "deit_bupt_balancedface_fr_random_224_"
        "bs256_adamw_actualLR1e-4_wu3_cosine_"
        "wd005_dp01_s32_m03_fp32"
    )

    output_dir = (
        "./pretraining/checkpoints/"
        "deit_bupt_balancedface_fr_random_224_"
        "bs256_adamw_actualLR1e-4_wu3_cosine_"
        "wd005_dp01_s32_m03_fp32"
    )