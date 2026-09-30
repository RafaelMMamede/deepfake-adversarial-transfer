import os

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

    # Dataset index cache
    cache_dir = "./pretraining/cache"
    use_cache = True
    rebuild_cache = False

    # ============================================================
    # Model
    # ============================================================
    network = "xception"
    mode = "normal"
    inc = 3

    # Only used to instantiate the detector class.
    # Ignored during FR pretraining because the script uses .features().
    num_classes_for_dummy_head = 2

    # Random initialization -> FR pretraining
    initial_pretrained = None

    # Temporary FR embedding head
    embedding_size = 512
    embedding_dropout = 0.0

    # Detector classifier dropout; unused during FR pretraining.
    dropout = 0.0

    # ============================================================
    # FR loss
    # ============================================================
    loss = "ElasticArcFacePlus"

    s = 64.0
    m = 0.3
    std = 0.05

    # Only used if loss = "AdaFace"
    h = 0.333
    t_alpha = 0.01

    # ============================================================
    # Training
    # ============================================================
    batch_size = 256
    val_batch_size = 512

    # Base LR at batch size 512.
    # Actual LR in script = lr / 512 * batch_size.
    # With batch_size=256: actual LR = 0.01 / 512 * 256 = 0.005
    lr = 0.01

    momentum = 0.9
    weight_decay = 5e-4

    num_epoch = 30

    num_workers = 8
    prefetch_factor = 4

    grad_clip = 5.0
    use_amp = True
    log_interval = 100

    @staticmethod
    def lr_func(epoch):
        if epoch < 10:
            return 1.0
        elif epoch < 20:
            return 0.1
        elif epoch < 26:
            return 0.01
        else:
            return 0.001

    # ============================================================
    # Logging / outputs
    # ============================================================
    use_wandb = False
    wandb_project = "fr_pretrain_deepfake_backbones"

    wandb_job_name = "xception_bupt_balancedface_fr_random_224_bs256_lr0005_m03"
    output_dir = "./pretraining/checkpoints/xception_bupt_balancedface_fr_random_224_bs256_lr0005_m03"