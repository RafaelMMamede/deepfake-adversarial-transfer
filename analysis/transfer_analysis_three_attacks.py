"""Three-attack extension of Rafael Mamede's transfer analysis.

Statistical estimators and node-jackknife implementation originate from:
https://github.com/RafaelMMamede/deepfake-adversarial-transfer
analysis/transfer_analysis_node_jackknife.py, blob
8ebd5b0c6623c7a2e4961308b7ae0eb2e3006169 (CC BY-NC 4.0).

All ASR values and statistical effects are stored on the unit interval.
Presentation functions convert them to percentages / percentage points.
"""
import hashlib
import json
import re
import warnings
from pathlib import Path

ATTACK_ORDER = ["aa", "cw_no_eot", "cw_eot"]
ATTACK_LABELS = {"aa": "AA", "cw_no_eot": "CW without EOT", "cw_eot": "CW-EOT"}
ATTACK_COMPARISONS = [
    ("cw_eot", "aa"),
    ("cw_eot", "cw_no_eot"),
    ("cw_no_eot", "aa"),
]
COMPATIBILITY_TESTS = [
    "same_backbone", "same_pretrain", "same_train_family_excl_ALL",
    "same_arch_family_excl_same_backbone",
]
DIRECTION_SPECS = [
    ("direction_transformer_to_cnn_vs_reverse", "arch_direction",
     "transformer->cnn", "cnn->transformer"),
    ("direction_fr_to_imgnet_vs_reverse", "pretrain_direction",
     "fr->imgnet", "imgnet->fr"),
    ("direction_all_to_single_vs_reverse", "training_direction",
     "ALL->single", "single->ALL"),
]
TEST_LABELS = {
    "same_backbone": "Shared backbone",
    "same_pretrain": "Shared pretraining",
    "same_train_family_excl_ALL": "Shared training data (excluding ALL)",
    "same_arch_family_excl_same_backbone": "Shared architecture family (different backbones)",
}
BACKBONE_ORDER = ["xception", "resnet34", "efficientnetb4", "deit", "vit", "swin"]
BACKBONE_LABELS = {
    "xception": "Xception", "resnet34": "ResNet34", "efficientnetb4": "EffNet-B4",
    "deit": "DeiT-S", "vit": "ViT-B/16", "swin": "Swin-T",
}
PRETRAIN_ORDER = ["imgnet", "fr"]
TRAIN_FAMILY_ORDER = ["ALL", "FS", "FR", "EFS", "FE"]
ARCH_ORDER = ["cnn", "transformer"]
SCHEMA_VERSION = "three-attacks-v2-oracles"


import os
import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# 0. Main entry point
# ============================================================

def run_transfer_analysis(
    df_transfer,
    attack_type=None,
    out_dir="transfer_analysis_node_jackknife",
    alpha=0.05,
    exclude_clip=True,
):
    """
    End-to-end transfer analysis.

    If exclude_clip=True:
      - removes CLIP origins and CLIP targets;
      - runs diagnostics, summaries, matrices, and tests only on non-CLIP models.

    Required columns:
      target_model, origin_model, asr, clean_acc,
      num_samples, num_clean_correct, num_success

    Recommended optional columns:
      avg_prob_drop_clean_correct, attack_name, attack_path
    """

    os.makedirs(out_dir, exist_ok=True)

    df = prepare_transfer_df(df_transfer, attack_type=attack_type)

    if exclude_clip:
        before = len(df)
        df = remove_clip_rows(df)
        after = len(df)

        print("=" * 80)
        print("CLIP EXCLUSION")
        print("=" * 80)
        print(f"Rows before CLIP exclusion: {before}")
        print(f"Rows after CLIP exclusion:  {after}")
        print(f"Rows removed:               {before - after}")

    print_basic_diagnostics(df)

    # --------------------------------------------------------
    # White-box validity table
    # --------------------------------------------------------
    whitebox_table = compute_whitebox_table(df)

    print("\n=== White-box source attack table ===")
    print(format_df(
        whitebox_table.sort_values(["clean_acc", "whitebox_asr"], ascending=False),
        float_cols=[
            "clean_acc",
            "whitebox_asr",
            "avg_prob_drop_clean_correct",
            "clean_acc_pct",
            "whitebox_asr_pct",
        ],
    ))

    # --------------------------------------------------------
    # Descriptive summaries
    # --------------------------------------------------------
    summaries = compute_all_descriptive_summaries(df)

    print("\n=== Overall summary ===")
    print(format_df(summaries["overall"]))

    print("\n=== White-box vs black-box summary ===")
    print(format_df(summaries["wb_bb"]))

    print("\n=== Black-box ASR by shared pretraining ===")
    print(format_df(summaries["same_pretrain"]))

    print("\n=== Black-box ASR by shared backbone ===")
    print(format_df(summaries["same_backbone"]))

    print("\n=== Black-box ASR by shared train family ===")
    print(format_df(summaries["same_train_family"]))

    # --------------------------------------------------------
    # Matrices
    # --------------------------------------------------------
    matrices = compute_transfer_matrices(df)

    print("\n=== Black-box ASR (%) by origin/target pretraining ===")
    print(matrices["pretrain"].round(2).to_string())

    print("\n=== Black-box ASR (%) by origin/target backbone ===")
    print(matrices["backbone"].round(2).to_string())

    print("\n=== Black-box ASR (%) by origin/target train family ===")
    print(matrices["train_family"].round(2).to_string())

    # --------------------------------------------------------
    # Statistical tests
    # --------------------------------------------------------
    stats_summary, stats_diffs, jackknife_replicates = run_all_matched_tests(
        df,
        alpha=alpha,
    )

    print("\n=== Matched statistical tests ===")
    print(format_df(
        stats_summary,
        float_cols=[
            "mean_true",
            "mean_false",
            "mean_diff",
            "ci_low",
            "ci_high",
            "se_node_jackknife",
            "p_node_jackknife",
            "p_holm",
        ],
    ))

    # --------------------------------------------------------
    # Save everything
    # --------------------------------------------------------
    save_outputs(
        out_dir=out_dir,
        df=df,
        whitebox_table=whitebox_table,
        summaries=summaries,
        matrices=matrices,
        stats_summary=stats_summary,
        stats_diffs=stats_diffs,
        jackknife_replicates=jackknife_replicates,
    )

    print(f"\nSaved all outputs to: {out_dir}")

    return {
        "df": df,
        "whitebox_table": whitebox_table,
        "summaries": summaries,
        "matrices": matrices,
        "stats_summary": stats_summary,
        "stats_diffs": stats_diffs,
        "jackknife_replicates": jackknife_replicates,
    }


# ============================================================
# 1. Data preparation
# ============================================================

def prepare_transfer_df(df_transfer, attack_type=None):
    df = df_transfer.copy()
    derived = [f"{role}_{field}" for role in ["origin", "target"]
               for field in ["pretrain", "backbone", "train_family", "arch_family"]]
    df = df.drop(columns=derived, errors="ignore")

    required_cols = [
        "target_model",
        "origin_model",
        "asr",
        "clean_acc",
        "num_samples",
        "num_clean_correct",
        "num_success",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = [
        "asr",
        "clean_acc",
        "num_samples",
        "num_clean_correct",
        "num_success",
        "adv_acc",
        "num_adv_correct",
        "avg_prob_drop_clean_correct",
        "median_prob_drop_clean_correct",
        "avg_prob_drop_top10_conf",
        "median_prob_drop_top10_conf",
        "avg_prob_drop_all",
        "median_prob_drop_all",
        "avg_prob_drop_success",
        "median_prob_drop_success",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[df["asr"].notna()].copy()

    if "attack_type" not in df.columns:
        if attack_type is not None:
            df["attack_type"] = attack_type
        else:
            df["attack_type"] = infer_attack_type(df)

    # Some evaluation CSVs store only the bare source name (for example,
    # ``deit_ALL``) and encode source pretraining in ``attack_path``. A node
    # jackknife requires the source and target endpoints to use the same
    # canonical detector IDs, so recover the source prefix before parsing
    # metadata or identifying white-box pairs.
    df["origin_model"] = canonicalize_origin_models(df)

    origin_meta = df["origin_model"].apply(parse_model_name).add_prefix("origin_")
    target_meta = df["target_model"].apply(parse_model_name).add_prefix("target_")

    df = pd.concat([df, origin_meta, target_meta], axis=1)

    df["same_pretrain"] = df["origin_pretrain"] == df["target_pretrain"]
    df["same_backbone"] = df["origin_backbone"] == df["target_backbone"]
    df["same_train_family"] = df["origin_train_family"] == df["target_train_family"]

    transformer_backbones = {
        "deit_small_patch16_224", "deit", "vit_base_patch16_224", "vit",
        "swin_tiny_patch4_window7_224", "swin",
    }
    df["origin_arch_family"] = np.where(
        df["origin_backbone"].isin(transformer_backbones), "transformer", "cnn"
    )
    df["target_arch_family"] = np.where(
        df["target_backbone"].isin(transformer_backbones), "transformer", "cnn"
    )
    df["same_arch_family"] = (
        df["origin_arch_family"] == df["target_arch_family"]
    )

    df["is_whitebox"] = df["origin_model"] == df["target_model"]
    df["is_blackbox"] = ~df["is_whitebox"]

    return df


def remove_clip_rows(df):
    """
    Remove CLIP models as origins and targets.

    This is the controlled comparison setting:
      - no CLIP source models;
      - no CLIP target models;
      - no rows where CLIP appears as pretraining or backbone.
    """

    d = df.copy()

    d = d[
        (~d["origin_pretrain"].eq("clip")) &
        (~d["target_pretrain"].eq("clip")) &
        (~d["origin_backbone"].eq("clip")) &
        (~d["target_backbone"].eq("clip")) &
        (~d["origin_model"].astype(str).str.lower().str.startswith("clip")) &
        (~d["target_model"].astype(str).str.lower().str.startswith("clip"))
    ].copy()

    return d


def infer_attack_type(df):
    """Infer AA/CW-EOT separately for every row."""
    text = pd.Series("", index=df.index, dtype="object")

    for col in ["attack_name", "attack_path"]:
        if col in df.columns:
            text = text + " " + df[col].fillna("").astype(str).str.lower()

    attack = pd.Series("unknown", index=df.index, dtype="object")
    attack[text.str.contains(r"cw[_-]?eot|\beot\b", regex=True)] = "cw_eot"
    attack[
        text.str.contains(r"autoattack|aa[_-]th[_-]aware", regex=True)
        & attack.eq("unknown")
    ] = "aa"
    return attack


def canonicalize_origin_models(df):
    """
    Return source IDs in the same ``pretrain_backbone_family`` form as targets.

    Already-canonical IDs are preserved. For bare source IDs, pretraining is
    recovered from ``attack_path``. Unknown paths fail loudly because merging
    two source checkpoints under one node ID invalidates node-level inference.
    """
    origin = df["origin_model"].fillna("").astype(str).str.strip()
    already_canonical = origin.str.startswith(
        ("imgnet_", "fr_", "clip_", "nopretrain_", "no_pretrain_")
    )

    if already_canonical.all():
        return origin

    if "attack_path" not in df.columns:
        examples = origin.loc[~already_canonical].drop_duplicates().head(5).tolist()
        raise ValueError(
            "Bare origin_model IDs require attack_path to recover pretraining. "
            f"Examples: {examples}"
        )

    path = df["attack_path"].fillna("").astype(str).str.lower()
    prefix = pd.Series("", index=df.index, dtype="object")
    prefix[path.str.contains(r"/fr_pretrain/", regex=True)] = "fr_"
    prefix[path.str.contains(r"/imgnet/", regex=True)] = "imgnet_"
    prefix[path.str.contains(r"/aa_th_aware/adv/adv/", regex=True)] = "imgnet_"
    prefix[path.str.contains(r"/(?:no_pretrain|nopretrain)/", regex=True)] = (
        "nopretrain_"
    )

    unresolved = ~already_canonical & prefix.eq("")
    if unresolved.any():
        examples = (
            df.loc[unresolved, ["origin_model", "attack_path"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise ValueError(
            "Could not recover source pretraining from attack_path. "
            f"Examples: {examples}"
        )

    canonical = origin.copy()
    canonical.loc[~already_canonical] = (
        prefix.loc[~already_canonical] + origin.loc[~already_canonical]
    )
    return canonical


def parse_model_name(name):
    """
    Expected:
      imgnet_xception_ALL
      imgnet_efficientnetb4_FS
      fr_resnet34_EFS
      fr_vit_FE
      clip_ALL
    """

    s = str(name).strip()
    parts = s.split("_")

    first = parts[0].lower()

    if first == "imgnet":
        pretrain = "imgnet"
        train_family = parts[-1]
        backbone = "_".join(parts[1:-1])

    elif first == "fr":
        pretrain = "fr"
        train_family = parts[-1]
        backbone = "_".join(parts[1:-1])

    elif first == "clip":
        pretrain = "clip"
        backbone = "clip"
        train_family = parts[-1]

    elif first in ["nopretrain", "no_pretrain"]:
        pretrain = "nopretrain"
        train_family = parts[-1]
        backbone = "_".join(parts[1:-1])

    else:
        pretrain = "unknown"
        train_family = parts[-1]
        backbone = "_".join(parts[:-1])

    return pd.Series({
        "pretrain": pretrain,
        "backbone": backbone,
        "train_family": train_family,
    })


# ============================================================
# 2. Diagnostics
# ============================================================

def print_basic_diagnostics(df):
    print("=" * 80)
    print("BASIC DIAGNOSTICS")
    print("=" * 80)

    print(f"Rows:        {len(df)}")
    print(f"Origins:     {df['origin_model'].nunique()}")
    print(f"Targets:     {df['target_model'].nunique()}")
    print(f"White-box:   {df['is_whitebox'].sum()}")
    print(f"Black-box:   {df['is_blackbox'].sum()}")

    print("\nAttack type counts:")
    print(df["attack_type"].value_counts(dropna=False).to_string())

    print("\nOrigin pretraining counts:")
    print(df["origin_pretrain"].value_counts(dropna=False).to_string())

    print("\nTarget pretraining counts:")
    print(df["target_pretrain"].value_counts(dropna=False).to_string())

    print("\nOrigin backbone counts:")
    print(df["origin_backbone"].value_counts(dropna=False).to_string())

    print("\nTarget backbone counts:")
    print(df["target_backbone"].value_counts(dropna=False).to_string())

    missing_diag = find_missing_diagonals(df)

    if len(missing_diag) > 0:
        print("\nWARNING: Missing white-box diagonal models:")
        for m in missing_diag:
            print(f"  {m}")
    else:
        print("\nAll common models have white-box diagonal rows.")


def find_missing_diagonals(df):
    all_models = sorted(set(df["origin_model"]).union(set(df["target_model"])))
    missing = []

    for m in all_models:
        has_diag = (
            (df["origin_model"] == m) &
            (df["target_model"] == m)
        ).any()

        if not has_diag:
            missing.append(m)

    return missing


# ============================================================
# 3. White-box table
# ============================================================

def compute_whitebox_table(df):
    diag = df[df["is_whitebox"]].copy()

    if len(diag) == 0:
        return pd.DataFrame()

    agg_dict = {
        "num_samples": "mean",
        "num_clean_correct": "mean",
        "num_success": "mean",
        "clean_acc": "mean",
        "asr": "mean",
    }

    if "avg_prob_drop_clean_correct" in diag.columns:
        agg_dict["avg_prob_drop_clean_correct"] = "mean"

    wb = (
        diag
        .groupby(["attack_type", "target_model"], as_index=False)
        .agg(agg_dict)
        .rename(columns={
            "target_model": "model",
            "asr": "whitebox_asr",
        })
    )

    for col in ["num_samples", "num_clean_correct", "num_success"]:
        wb[col] = wb[col].round().astype(int)

    if "avg_prob_drop_clean_correct" not in wb.columns:
        wb["avg_prob_drop_clean_correct"] = np.nan

    meta = wb["model"].apply(parse_model_name)
    wb = pd.concat([wb, meta], axis=1)

    wb["clean_acc_pct"] = wb["clean_acc"] * 100
    wb["whitebox_asr_pct"] = wb["whitebox_asr"] * 100

    return wb


# ============================================================
# 4. Descriptive summaries
# ============================================================

def summarize_group(group):
    out = {
        "rows": len(group),
        "num_origins": group["origin_model"].nunique(),
        "num_targets": group["target_model"].nunique(),
        "mean_asr": group["asr"].mean(),
        "median_asr": group["asr"].median(),
        "std_asr": group["asr"].std(),
        "mean_clean_acc": group["clean_acc"].mean(),
        "median_clean_acc": group["clean_acc"].median(),
        "mean_num_clean_correct": group["num_clean_correct"].mean(),
        "median_num_clean_correct": group["num_clean_correct"].median(),
    }

    if "avg_prob_drop_clean_correct" in group.columns:
        out["mean_prob_drop_clean_correct"] = group["avg_prob_drop_clean_correct"].mean()
        out["median_prob_drop_clean_correct"] = group["avg_prob_drop_clean_correct"].median()

    return pd.Series(out)


def grouped_summary(df, group_cols):
    if len(df) == 0:
        return pd.DataFrame()

    return (
        df
        .groupby(group_cols, dropna=False)
        .apply(summarize_group)
        .reset_index()
    )


def compute_all_descriptive_summaries(df):
    df_bb = df[df["is_blackbox"]].copy()

    summaries = {}

    summaries["overall"] = grouped_summary(df, ["attack_type"])

    summaries["wb_bb"] = grouped_summary(df, ["attack_type", "is_whitebox"])
    summaries["wb_bb"]["setting"] = np.where(
        summaries["wb_bb"]["is_whitebox"],
        "white-box",
        "black-box",
    )

    summaries["same_pretrain"] = grouped_summary(
        df_bb,
        ["attack_type", "same_pretrain"],
    )

    summaries["same_backbone"] = grouped_summary(
        df_bb,
        ["attack_type", "same_backbone"],
    )

    summaries["same_train_family"] = grouped_summary(
        df_bb,
        ["attack_type", "same_train_family"],
    )

    summaries["origin_pretrain"] = grouped_summary(
        df_bb,
        ["attack_type", "origin_pretrain"],
    )

    summaries["target_pretrain"] = grouped_summary(
        df_bb,
        ["attack_type", "target_pretrain"],
    )

    summaries["origin_backbone"] = grouped_summary(
        df_bb,
        ["attack_type", "origin_backbone"],
    )

    summaries["target_backbone"] = grouped_summary(
        df_bb,
        ["attack_type", "target_backbone"],
    )

    summaries["origin_train_family"] = grouped_summary(
        df_bb,
        ["attack_type", "origin_train_family"],
    )

    summaries["target_train_family"] = grouped_summary(
        df_bb,
        ["attack_type", "target_train_family"],
    )

    return summaries


# ============================================================
# 5. Matrices
# ============================================================

def compute_transfer_matrices(df):
    df_bb = df[df["is_blackbox"]].copy()

    matrices = {}

    matrices["pretrain"] = make_matrix(
        df_bb,
        index="target_pretrain",
        columns="origin_pretrain",
        values="asr",
        scale=100,
    )

    matrices["backbone"] = make_matrix(
        df_bb,
        index="target_backbone",
        columns="origin_backbone",
        values="asr",
        scale=100,
    )

    matrices["train_family"] = make_matrix(
        df_bb,
        index="target_train_family",
        columns="origin_train_family",
        values="asr",
        scale=100,
    )

    matrices["model"] = make_matrix(
        df_bb,
        index="target_model",
        columns="origin_model",
        values="asr",
        scale=100,
    )

    return matrices


def make_matrix(df, index, columns, values="asr", scale=1):
    return (
        df
        .pivot_table(
            index=index,
            columns=columns,
            values=values,
            aggfunc="mean",
        )
        * scale
    )


# ============================================================
# 6. Statistical tests
# ============================================================

def node_jackknife_inference(
    df,
    estimator,
    theta_hat=None,
    origin_col="origin_model",
    target_col="target_model",
    alpha=0.05,
):
    """
    Leave-one-node-out inference for a scalar directed-dyadic estimator.

    For node g, every ordered dyad with g as source OR target is removed and
    the complete estimator is rerun. Direction is retained for all remaining
    dyads. The function returns a standard delete-one-node jackknife variance,
    an asymptotic-normal confidence interval, and an approximate two-sided
    Wald p-value for H0: theta = 0.

    The jackknife estimates the variance. Confidence intervals and p-values
    additionally assume that the estimator standardized by its jackknife
    standard error is approximately N(0, 1). The procedure does not produce
    an exact randomization p-value.
    """
    if origin_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"Expected node columns {origin_col!r} and {target_col!r}."
        )

    if theta_hat is None:
        theta_hat = estimator(df)
    theta_hat = float(theta_hat)

    if not np.isfinite(theta_hat):
        raise ValueError("The full-sample estimator is not finite.")

    nodes = sorted(
        set(df[origin_col].dropna().astype(str))
        | set(df[target_col].dropna().astype(str))
    )
    if len(nodes) < 4:
        raise ValueError("At least four detector nodes are required.")

    rows = []
    for node in nodes:
        reduced = df[
            df[origin_col].astype(str).ne(node)
            & df[target_col].astype(str).ne(node)
        ].copy()
        estimate = float(estimator(reduced))

        if not np.isfinite(estimate):
            raise RuntimeError(
                "The effect became non-estimable after removing "
                f"detector node {node!r}."
            )

        rows.append({
            "omitted_node": node,
            "estimate": estimate,
            "num_rows_remaining": len(reduced),
        })

    replicates = pd.DataFrame(rows)
    loo = replicates["estimate"].to_numpy(dtype=float)
    G = len(loo)
    loo_mean = float(loo.mean())

    # Standard delete-one jackknife scaling, applied at the detector-node
    # rather than dyad/stratum level.
    variance_multiplier = (G - 1) / G
    variance = variance_multiplier * np.sum((loo - loo_mean) ** 2)
    variance = float(max(variance, 0.0))
    se = float(np.sqrt(variance))

    critical = float(stats.norm.ppf(1 - alpha / 2))
    ci_low = float(theta_hat - critical * se)
    ci_high = float(theta_hat + critical * se)

    if se == 0.0:
        z_stat = np.inf if theta_hat != 0.0 else 0.0
        p_value = 0.0 if theta_hat != 0.0 else 1.0
    else:
        z_stat = float(theta_hat / se)
        p_value = float(2 * stats.norm.sf(abs(z_stat)))

    replicates["change_from_full"] = replicates["estimate"] - theta_hat
    influence_idx = replicates["change_from_full"].abs().idxmax()

    result = {
        "num_nodes": G,
        "se_node_jackknife": se,
        "variance_node_jackknife": variance,
        "jackknife_variance_multiplier": variance_multiplier,
        "reference_distribution": "standard_normal",
        "z_node_jackknife": z_stat,
        "p_node_jackknife": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "loo_mean": loo_mean,
        "loo_min": float(loo.min()),
        "loo_max": float(loo.max()),
        "max_abs_loo_change": float(
            replicates["change_from_full"].abs().max()
        ),
        "most_influential_node": str(
            replicates.loc[influence_idx, "omitted_node"]
        ),
    }
    return result, replicates


def matched_effect_point_estimate(
    df,
    effect_col,
    strata_cols,
    outcome_col="asr",
):
    """Compute the matched macro-average without making independence claims."""
    d = df[
        df[effect_col].isin([True, False]) & df[outcome_col].notna()
    ]
    grouped = d.groupby(
        strata_cols + [effect_col],
        dropna=False,
        observed=False,
    )[outcome_col]
    means = grouped.mean().unstack(effect_col)
    counts = grouped.size().unstack(effect_col)

    if True not in means.columns or False not in means.columns:
        return make_empty_test_result(effect_col), pd.DataFrame()

    eligible = means[True].notna() & means[False].notna()
    if not eligible.any():
        return make_empty_test_result(effect_col), pd.DataFrame()

    means = means.loc[eligible]
    counts = counts.loc[eligible].fillna(0)
    diffs = means.index.to_frame(index=False)
    diffs["effect"] = effect_col
    diffs["mean_true"] = means[True].to_numpy()
    diffs["mean_false"] = means[False].to_numpy()
    diffs["diff"] = diffs["mean_true"] - diffs["mean_false"]
    diffs["n_true_rows"] = counts[True].to_numpy(dtype=int)
    diffs["n_false_rows"] = counts[False].to_numpy(dtype=int)
    diffs["n_rows"] = diffs["n_true_rows"] + diffs["n_false_rows"]

    result = {
        "effect": effect_col,
        "n_strata": len(diffs),
        "mean_true": float(diffs["mean_true"].mean()),
        "mean_false": float(diffs["mean_false"].mean()),
        "mean_diff": float(diffs["diff"].mean()),
    }
    return result, diffs


def matched_effect_test(
    df,
    effect_col,
    strata_cols,
    outcome_col="asr",
    alpha=0.05,
):
    result, diffs = matched_effect_point_estimate(
        df=df,
        effect_col=effect_col,
        strata_cols=strata_cols,
        outcome_col=outcome_col,
    )
    if diffs.empty:
        return result, diffs, pd.DataFrame()

    def estimator(reduced):
        estimate, _ = matched_effect_point_estimate(
            df=reduced,
            effect_col=effect_col,
            strata_cols=strata_cols,
            outcome_col=outcome_col,
        )
        return estimate["mean_diff"]

    inference, replicates = node_jackknife_inference(
        df=df,
        estimator=estimator,
        theta_hat=result["mean_diff"],
        alpha=alpha,
    )
    result.update(inference)
    return result, diffs, replicates


def attack_compatibility_interaction_point_estimate(
    df,
    effect_col,
    strata_cols,
    attack_a="cw_eot",
    attack_b="aa",
    outcome_col="asr",
):
    """
    Estimate whether a matched compatibility effect differs between attacks.

    The estimand is

        Delta(attack_a) - Delta(attack_b),

    where each Delta is the matched same-versus-different effect returned by
    ``matched_effect_point_estimate``. Only strata eligible under both attacks
    enter the interaction, so the attack effects are compared over the same
    matched configurations.

    In the returned summary, ``mean_true`` is Delta(attack_a),
    ``mean_false`` is Delta(attack_b), and ``mean_diff`` is their difference.
    """
    required = {"attack_type", effect_col, outcome_col, *strata_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Missing columns for attack-by-compatibility interaction: "
            f"{missing}"
        )

    attack_results = {}
    attack_diffs = {}

    for attack in (attack_a, attack_b):
        local = df[df["attack_type"].eq(attack)].copy()
        result, diffs = matched_effect_point_estimate(
            df=local,
            effect_col=effect_col,
            strata_cols=strata_cols,
            outcome_col=outcome_col,
        )
        attack_results[attack] = result
        attack_diffs[attack] = diffs

    if any(attack_diffs[a].empty for a in (attack_a, attack_b)):
        result = make_empty_test_result(
            f"{attack_a}_minus_{attack_b}__{effect_col}"
        )
        result.update({
            "n_strata_attack_a": len(attack_diffs[attack_a]),
            "n_strata_attack_b": len(attack_diffs[attack_b]),
        })
        return result, pd.DataFrame()

    value_cols = ["mean_true", "mean_false", "diff"]
    left = attack_diffs[attack_a][strata_cols + value_cols].rename(
        columns={c: f"{c}_{attack_a}" for c in value_cols}
    )
    right = attack_diffs[attack_b][strata_cols + value_cols].rename(
        columns={c: f"{c}_{attack_b}" for c in value_cols}
    )

    paired = left.merge(
        right,
        on=strata_cols,
        how="inner",
        validate="one_to_one",
    )

    if paired.empty:
        result = make_empty_test_result(
            f"{attack_a}_minus_{attack_b}__{effect_col}"
        )
        result.update({
            "n_strata_attack_a": len(attack_diffs[attack_a]),
            "n_strata_attack_b": len(attack_diffs[attack_b]),
        })
        return result, paired

    paired[f"compatibility_effect_{attack_a}"] = paired[f"diff_{attack_a}"]
    paired[f"compatibility_effect_{attack_b}"] = paired[f"diff_{attack_b}"]
    paired["interaction_diff"] = (
        paired[f"compatibility_effect_{attack_a}"]
        - paired[f"compatibility_effect_{attack_b}"]
    )

    effect_a = float(paired[f"compatibility_effect_{attack_a}"].mean())
    effect_b = float(paired[f"compatibility_effect_{attack_b}"].mean())

    result = {
        "effect": f"Delta_{attack_a} - Delta_{attack_b}",
        "n_strata": len(paired),
        "n_strata_attack_a": len(attack_diffs[attack_a]),
        "n_strata_attack_b": len(attack_diffs[attack_b]),
        "mean_true": effect_a,
        "mean_false": effect_b,
        "mean_diff": float(effect_a - effect_b),
    }
    return result, paired


def attack_compatibility_interaction_test(
    df,
    effect_col,
    strata_cols,
    attack_a="cw_eot",
    attack_b="aa",
    outcome_col="asr",
    alpha=0.05,
):
    """
    Node-jackknife test of an attack-by-compatibility interaction.

    Every leave-one-node replicate removes the detector from both source and
    target roles, reconstructs the matched strata for both attacks, and then
    recomputes Delta(attack_a) - Delta(attack_b).
    """
    analysis_df = df[
        df["attack_type"].isin([attack_a, attack_b])
    ].copy()

    result, paired_diffs = (
        attack_compatibility_interaction_point_estimate(
            df=analysis_df,
            effect_col=effect_col,
            strata_cols=strata_cols,
            attack_a=attack_a,
            attack_b=attack_b,
            outcome_col=outcome_col,
        )
    )
    if paired_diffs.empty:
        return result, paired_diffs, pd.DataFrame()

    def estimator(reduced):
        estimate, _ = attack_compatibility_interaction_point_estimate(
            df=reduced,
            effect_col=effect_col,
            strata_cols=strata_cols,
            attack_a=attack_a,
            attack_b=attack_b,
            outcome_col=outcome_col,
        )
        return estimate["mean_diff"]

    inference, replicates = node_jackknife_inference(
        df=analysis_df,
        estimator=estimator,
        theta_hat=result["mean_diff"],
        alpha=alpha,
    )
    result.update(inference)
    return result, paired_diffs, replicates


def make_empty_test_result(effect_col):
    return {
        "effect": effect_col,
        "n_strata": 0,
        "mean_true": np.nan,
        "mean_false": np.nan,
        "mean_diff": np.nan,
        "num_nodes": 0,
        "se_node_jackknife": np.nan,
        "variance_node_jackknife": np.nan,
        "jackknife_variance_multiplier": np.nan,
        "reference_distribution": "standard_normal",
        "z_node_jackknife": np.nan,
        "p_node_jackknife": np.nan,
        "ci_low": np.nan,
        "ci_high": np.nan,
        "loo_mean": np.nan,
        "loo_min": np.nan,
        "loo_max": np.nan,
        "max_abs_loo_change": np.nan,
        "most_influential_node": np.nan,
    }


def labelled_contrast_point_estimate(
    df,
    label_col,
    level_a,
    level_b,
    strata_cols,
    test_name,
    outcome_col="asr",
):
    """Compute a matched level_a - level_b macro-average."""
    d = df[df[label_col].isin([level_a, level_b])].copy()
    grouped = (
        d.groupby(strata_cols + [label_col], dropna=False)[outcome_col]
        .mean()
        .unstack(label_col)
    )
    if level_a not in grouped.columns or level_b not in grouped.columns:
        return make_empty_test_result(test_name), pd.DataFrame(), d

    grouped = grouped.dropna(subset=[level_a, level_b]).reset_index()
    if grouped.empty:
        return make_empty_test_result(test_name), grouped, d

    grouped = grouped.rename(
        columns={level_a: "mean_true", level_b: "mean_false"}
    )
    grouped["diff"] = grouped["mean_true"] - grouped["mean_false"]
    result = {
        "effect": f"{level_a} - {level_b}",
        "n_strata": len(grouped),
        "mean_true": float(grouped["mean_true"].mean()),
        "mean_false": float(grouped["mean_false"].mean()),
        "mean_diff": float(grouped["diff"].mean()),
    }
    return result, grouped, d


def labelled_contrast_test(
    df, label_col, level_a, level_b, strata_cols, test_name,
    outcome_col="asr", alpha=0.05,
):
    result, grouped, analysis_df = labelled_contrast_point_estimate(
        df=df,
        label_col=label_col,
        level_a=level_a,
        level_b=level_b,
        strata_cols=strata_cols,
        test_name=test_name,
        outcome_col=outcome_col,
    )
    if grouped.empty:
        return result, grouped, pd.DataFrame()

    def estimator(reduced):
        estimate, _, _ = labelled_contrast_point_estimate(
            df=reduced,
            label_col=label_col,
            level_a=level_a,
            level_b=level_b,
            strata_cols=strata_cols,
            test_name=test_name,
            outcome_col=outcome_col,
        )
        return estimate["mean_diff"]

    inference, replicates = node_jackknife_inference(
        df=analysis_df,
        estimator=estimator,
        theta_hat=result["mean_diff"],
        alpha=alpha,
    )
    result.update(inference)
    return result, grouped, replicates


def add_reverse_pair_columns(df):
    """Annotate exact A<->B detector pairs and their meaningful directions."""
    d = df.copy()
    d["model_pair"] = d.apply(
        lambda r: " || ".join(sorted([str(r.origin_model), str(r.target_model)])), axis=1
    )
    d["arch_direction"] = d["origin_arch_family"] + "->" + d["target_arch_family"]
    d["pretrain_direction"] = d["origin_pretrain"] + "->" + d["target_pretrain"]
    d["training_direction"] = np.select(
        [
            d["origin_train_family"].eq("ALL") & ~d["target_train_family"].eq("ALL"),
            ~d["origin_train_family"].eq("ALL") & d["target_train_family"].eq("ALL"),
        ],
        ["ALL->single", "single->ALL"], default="other",
    )
    return d




def comparison_specs(df):
    bb = add_reverse_pair_columns(df[df["is_blackbox"] & df["asr"].notna()].copy())
    specs = [
        {"name": "same_pretrain", "effect_col": "same_pretrain",
         "strata_cols": ["attack_type", "origin_backbone", "target_backbone",
                         "origin_train_family", "target_train_family"], "df": bb},
        {"name": "same_backbone", "effect_col": "same_backbone",
         "strata_cols": ["attack_type", "origin_pretrain", "target_pretrain",
                         "origin_train_family", "target_train_family"], "df": bb},
        {"name": "same_train_family_excl_ALL", "effect_col": "same_train_family",
         "strata_cols": ["attack_type", "origin_pretrain", "target_pretrain",
                         "origin_backbone", "target_backbone"],
         "df": bb[~bb.origin_train_family.eq("ALL") & ~bb.target_train_family.eq("ALL")].copy()},
        {"name": "same_arch_family_excl_same_backbone", "effect_col": "same_arch_family",
         "strata_cols": ["attack_type", "origin_pretrain", "target_pretrain",
                         "origin_train_family", "target_train_family"],
         "df": bb[~bb.same_backbone].copy()},
    ]
    return bb, specs


def run_all_matched_tests(df, alpha=0.05, compute_inference=True):
    """All per-attack and pairwise contrasts; 36 primary tests for three attacks.

    The expanded family comprises 7 tests per attack (4 compatibility and
    3 directional) plus 5 per attack comparison (4 interactions and 1 overall).
    Pooled descriptive duplicates are excluded. The original 19-test AA /
    CW-EOT adjustment is also retained in separately named columns.
    """
    bb, specs = comparison_specs(df)
    observed = set(bb.attack_type.dropna().astype(str))
    attacks = [a for a in ATTACK_ORDER if a in observed]
    unknown = observed - set(ATTACK_ORDER)
    if unknown:
        raise ValueError(f"Unrecognized attack labels: {sorted(unknown)}")
    comparisons = [(a, b) for a, b in ATTACK_COMPARISONS if {a, b} <= observed]
    rows, diffs_out, jackknife_out = [], {}, {}

    def record(result, diffs, replicates, scope, attack, name, key):
        result = dict(result)
        result.update(scope=scope, attack_type=attack, test_name=name)
        rows.append(result)
        diffs_out[key] = diffs
        jackknife_out[key] = replicates

    def matched(data, effect, strata):
        if compute_inference:
            return matched_effect_test(data, effect, strata, alpha=alpha)
        r, d = matched_effect_point_estimate(data, effect, strata)
        return r, d, pd.DataFrame()

    def labelled(data, label, a, b, strata, name):
        if compute_inference:
            return labelled_contrast_test(data, label, a, b, strata, name, alpha=alpha)
        r, d, _ = labelled_contrast_point_estimate(data, label, a, b, strata, name)
        return r, d, pd.DataFrame()

    for spec in specs:
        r, d, j = matched(spec["df"], spec["effect_col"], spec["strata_cols"])
        record(r, d, j, "all_attacks", "all", spec["name"], "all_attacks__"+spec["name"])

    for attack in attacks:
        for spec in specs:
            local = spec["df"][spec["df"].attack_type.eq(attack)].copy()
            strata = [c for c in spec["strata_cols"] if c != "attack_type"]
            r, d, j = matched(local, spec["effect_col"], strata)
            record(r, d, j, "per_attack", attack, spec["name"], attack+"__"+spec["name"])
        for name, label, a, b in DIRECTION_SPECS:
            local = bb[bb.attack_type.eq(attack)].copy()
            r, d, j = labelled(local, label, a, b, ["model_pair"], name)
            record(r, d, j, "per_attack", attack, name, attack+"__"+name)

    for name, label, a, b in DIRECTION_SPECS:
        r, d, j = labelled(bb, label, a, b, ["attack_type", "model_pair"], name)
        record(r, d, j, "per_attack", "all", name, name)

    for attack_a, attack_b in comparisons:
        contrast = f"{attack_a}_minus_{attack_b}"
        for spec in specs:
            data = spec["df"][spec["df"].attack_type.isin([attack_a, attack_b])].copy()
            strata = [c for c in spec["strata_cols"] if c != "attack_type"]
            if compute_inference:
                r, d, j = attack_compatibility_interaction_test(
                    data, spec["effect_col"], strata, attack_a=attack_a,
                    attack_b=attack_b, alpha=alpha)
            else:
                r, d = attack_compatibility_interaction_point_estimate(
                    data, spec["effect_col"], strata, attack_a=attack_a, attack_b=attack_b)
                j = pd.DataFrame()
            name = "attack_interaction_"+spec["name"]
            record(r, d, j, "attack_interaction", contrast, name, contrast+"__"+name)

        name = f"attack_{attack_a}_vs_{attack_b}"
        r, d, j = labelled(bb, "attack_type", attack_a, attack_b,
                          ["origin_model", "target_model"], name)
        record(r, d, j, "cross_attack", contrast, name, name)

    out = pd.DataFrame(rows)
    inference_cols = [
        "n_strata_attack_a", "n_strata_attack_b", "num_nodes",
        "se_node_jackknife", "variance_node_jackknife", "jackknife_variance_multiplier",
        "reference_distribution", "z_node_jackknife", "ci_low", "ci_high",
        "p_node_jackknife", "loo_mean", "loo_min", "loo_max",
        "max_abs_loo_change", "most_influential_node",
    ]
    for col in inference_cols:
        if col not in out:
            out[col] = np.nan
    out = add_holm_correction(out, alpha)
    return out, diffs_out, jackknife_out


def _holm_values(pvalues, family_size):
    """Keep non-estimable hypotheses in the planned family as p=1."""
    p = pd.to_numeric(pvalues, errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(p)
    if family_size < len(p):
        raise ValueError("Family size cannot be smaller than its hypotheses.")
    expanded = np.r_[np.where(finite, p, 1.0), np.ones(family_size-len(p))]
    order = np.argsort(expanded, kind="stable")
    adjusted = np.maximum.accumulate((family_size-np.arange(family_size))*expanded[order])
    unsorted = np.empty(family_size)
    unsorted[order] = np.minimum(adjusted, 1.0)
    answer = unsorted[:len(p)]
    answer[~finite] = np.nan
    return answer


def add_holm_correction(stats_summary, alpha=0.05):
    out = stats_summary.copy()
    per_tests = set(COMPATIBILITY_TESTS) | {x[0] for x in DIRECTION_SPECS}
    interaction_tests = {"attack_interaction_"+x for x in COMPATIBILITY_TESTS}
    attacks = [a for a in ATTACK_ORDER if (
        out.scope.eq("per_attack") & out.attack_type.eq(a)).any()]
    pairs = [(a, b) for a,b in ATTACK_COMPARISONS if a in attacks and b in attacks]
    cross_names = {f"attack_{a}_vs_{b}" for a,b in pairs}
    contrast_names = {f"{a}_minus_{b}" for a,b in pairs}
    primary = (
        (out.scope.eq("per_attack") & out.attack_type.isin(attacks) & out.test_name.isin(per_tests))
        | (out.scope.eq("cross_attack") & out.test_name.isin(cross_names))
        | (out.scope.eq("attack_interaction") & out.attack_type.isin(contrast_names)
           & out.test_name.isin(interaction_tests))
    )
    m = 7*len(attacks)+5*len(pairs)
    out["primary_holm_family"] = primary
    out["holm_family_size"] = m
    out["p_holm"] = np.nan
    out["reject_holm"] = False
    if primary.sum() != m:
        raise AssertionError(f"Expected {m} hypotheses, found {primary.sum()}.")
    if m:
        out.loc[primary,"p_holm"] = _holm_values(out.loc[primary,"p_node_jackknife"],m)
        out.loc[primary,"reject_holm"] = out.loc[primary,"p_holm"].le(alpha)
    legacy = (
        (out.scope.eq("per_attack") & out.attack_type.isin(["aa","cw_eot"]) & out.test_name.isin(per_tests))
        | (out.scope.eq("cross_attack") & out.test_name.eq("attack_cw_eot_vs_aa"))
        | (out.scope.eq("attack_interaction") & out.attack_type.eq("cw_eot_minus_aa")
           & out.test_name.isin(interaction_tests))
    )
    legacy_complete = {"aa","cw_eot"} <= set(attacks)
    out["legacy_holm_family"] = legacy if legacy_complete else False
    out["legacy_holm_family_size"] = 19 if legacy_complete else 0
    out["p_holm_legacy"] = np.nan
    out["reject_holm_legacy"] = False
    if legacy_complete:
        if legacy.sum() != 19:
            raise AssertionError("Original AA / CW-EOT family must contain 19 tests.")
        out.loc[legacy,"p_holm_legacy"] = _holm_values(out.loc[legacy,"p_node_jackknife"],19)
        out.loc[legacy,"reject_holm_legacy"] = out.loc[legacy,"p_holm_legacy"].le(alpha)
    return out


def save_outputs(
    out_dir,
    df,
    whitebox_table,
    summaries,
    matrices,
    stats_summary,
    stats_diffs,
    jackknife_replicates,
):
    df.to_csv(f"{out_dir}/annotated_full_results.csv", index=False)
    whitebox_table.to_csv(f"{out_dir}/whitebox_source_table.csv", index=False)

    for name, table in summaries.items():
        table.to_csv(f"{out_dir}/summary_{name}.csv", index=False)

    for name, matrix in matrices.items():
        matrix.to_csv(f"{out_dir}/matrix_{name}_asr_pct.csv")

    stats_summary.to_csv(f"{out_dir}/matched_statistical_tests.csv", index=False)
    stats_summary[
        stats_summary["scope"].eq("attack_interaction")
    ].to_csv(
        f"{out_dir}/attack_compatibility_interactions.csv",
        index=False,
    )

    diffs_dir = f"{out_dir}/matched_diffs"
    os.makedirs(diffs_dir, exist_ok=True)

    for name, diffs in stats_diffs.items():
        diffs.to_csv(f"{diffs_dir}/{name}.csv", index=False)

    jackknife_dir = f"{out_dir}/node_jackknife_replicates"
    os.makedirs(jackknife_dir, exist_ok=True)

    for name, replicates in jackknife_replicates.items():
        replicates.to_csv(f"{jackknife_dir}/{name}.csv", index=False)


def format_df(df, float_cols=None):
    if df is None or len(df) == 0:
        return "<empty>"

    if float_cols is None:
        float_cols = [
            c for c in df.columns
            if pd.api.types.is_float_dtype(df[c])
        ]

    formatters = {
        c: "{:.4f}".format
        for c in float_cols
        if c in df.columns
    }

    return df.to_string(index=False, formatters=formatters)


# ============================================================
# 8. Example runs
# ============================================================



if __name__ == "__main__":
    print(
        "Import this module, label the AA, CW-no-EOT and CW-EOT frames row-wise, "
        "concatenate them, and call run_transfer_analysis(...)."
    )


def normalized_stem(path):
    return re.sub(r"\s*\(\d+\)$", "", Path(path).stem.lower().strip())


def resolve_input_file(data_dir, names, required=True, extra_dirs=()):
    """Look in the configured directory, its data/ child, then explicit extras.

    Copy suffixes such as '(1)' are accepted. Ambiguous different files in the
    same search directory fail instead of silently selecting a stale version.
    """
    if isinstance(names, str):
        names = [names]
    wanted = {normalized_stem(n) for n in names}
    bases = []
    for base in [Path(data_dir), Path(data_dir)/"data", Path(data_dir)/"analysis"/"data",
                 Path(data_dir)/"analysis", *map(Path,extra_dirs)]:
        if base not in bases:
            bases.append(base)
    for base in bases:
        if not base.is_dir():
            continue
        found = sorted([p for p in base.glob("*.csv") if normalized_stem(p) in wanted])
        if found:
            hashes = {hashlib.sha256(p.read_bytes()).hexdigest() for p in found}
            if len(hashes) != 1:
                raise ValueError(f"Ambiguous inputs in {base}: {[p.name for p in found]}. "
                                 "Keep the intended version or set DATA_DIR explicitly.")
            return min(found, key=lambda p:(p.name not in names,len(p.name),p.name)).resolve()
    if required:
        raise FileNotFoundError(
            f"Missing required input {names}. Searched: {[str(p.resolve()) for p in bases]}. "
            "Place the CSVs together or update DATA_DIR.")
    return None


def infer_attack_type(df):
    """Diagnostic inference; explicit labels supplied by load_transfer_results win."""
    text = pd.Series("", index=df.index, dtype="object")
    for col in ["attack_name","attack_path"]:
        if col in df:
            text = text+" "+df[col].fillna("").astype(str).str.lower()
    inferred = pd.Series("unknown", index=df.index, dtype="object")
    no_eot = text.str.contains(r"cw[_-](?:no[_-]eot|without[_-]eot)",regex=True)
    inferred.loc[no_eot] = "cw_no_eot"
    inferred.loc[~no_eot & text.str.contains(r"cw[_-]?eot|\beot\b",regex=True)] = "cw_eot"
    inferred.loc[inferred.eq("unknown") & text.str.contains(r"autoattack|aa[_-]th[_-]aware",regex=True)] = "aa"
    return inferred


def load_transfer_results(path, attack_type):
    if attack_type not in ATTACK_ORDER:
        raise ValueError(f"Unsupported explicit attack label: {attack_type}")
    raw = pd.read_csv(path)
    required = {"target_model","origin_model","asr","clean_acc",
                "num_samples","num_clean_correct","num_success"}
    if required-set(raw):
        raise ValueError(f"{path}: missing {sorted(required-set(raw))}")
    raw["origin_model_raw"] = raw["origin_model"]
    if "attack_path" in raw:
        raw["attack_path"] = raw["attack_path"].astype(str).str.replace("\\","/",regex=False)
    raw["origin_model"] = canonicalize_origin_models(raw)
    raw["attack_type"] = attack_type
    raw["input_file"] = str(Path(path).resolve())
    if raw[["asr","num_clean_correct","num_success"]].isna().any().any():
        raise ValueError(f"{path}: missing primary outcome or count values")
    for col in ["asr","clean_acc","num_samples","num_clean_correct","num_success"]:
        raw[col] = pd.to_numeric(raw[col],errors="raise")
    if not raw.asr.between(0,1).all():
        raise ValueError(f"{path}: ASR must be on [0,1]")
    if not ((raw.num_success >= 0) & (raw.num_success <= raw.num_clean_correct)
            & (raw.num_clean_correct > 0) & (raw.num_clean_correct <= raw.num_samples)).all():
        raise ValueError(f"{path}: invalid success / clean-correct / sample counts")
    if not np.allclose(raw.asr,raw.num_success/raw.num_clean_correct,rtol=0,atol=1e-10):
        raise ValueError(f"{path}: ASR disagrees with num_success / num_clean_correct")
    d = prepare_transfer_df(raw)
    if d.duplicated(["attack_type","origin_model","target_model"]).any():
        raise ValueError(f"{path}: duplicate source-target attack rows")
    return d


def validate_model_grid(df, expected_models=60):
    rows=[]
    for attack,d in df.groupby("attack_type",sort=False):
        sources=set(d.origin_model); targets=set(d.target_model)
        if sources != targets:
            raise ValueError(f"{attack}: source and target model pools differ")
        if len(sources) != expected_models or len(d) != expected_models**2:
            raise ValueError(f"{attack}: expected {expected_models}x{expected_models}, "
                             f"got {len(sources)} sources, {len(targets)} targets, {len(d)} rows")
        if d.duplicated(["origin_model","target_model"]).any():
            raise ValueError(f"{attack}: duplicate source-target pairs")
        unknown=d.origin_pretrain.eq("unknown") | d.target_pretrain.eq("unknown")
        if unknown.any():
            raise ValueError(f"{attack}: unknown detector metadata")
        for field in ["num_samples","num_clean_correct","target_threshold"]:
            if field in d and d.groupby("target_model")[field].nunique().gt(1).any():
                raise ValueError(f"{attack}: {field} varies across sources for the same target")
        rows.append({"attack_type":attack,"sources":len(sources),"targets":len(targets),
                     "rows":len(d),"whitebox_rows":int(d.is_whitebox.sum()),
                     "blackbox_rows":int(d.is_blackbox.sum()),
                     "sample_count_min":int(d.num_samples.min()),
                     "sample_count_max":int(d.num_samples.max())})
    return pd.DataFrame(rows)


def audit_attack_alignment(df):
    keys=["origin_model","target_model"]
    rows=[]; detail=[]
    available=set(df.attack_type)
    for a,b in ATTACK_COMPARISONS:
        if not {a,b} <= available:
            continue
        da=df[df.attack_type.eq(a)].set_index(keys)
        db=df[df.attack_type.eq(b)].set_index(keys)
        common=da.index.intersection(db.index)
        if len(common)!=len(da) or len(common)!=len(db):
            raise ValueError(f"{a} versus {b}: source-target coverage differs")
        row={"attack_a":a,"attack_b":b,"paired_rows":len(common)}
        for col in ["num_samples","num_clean_correct","target_threshold"]:
            if col not in da or col not in db:
                continue
            delta=da.loc[common,col]-db.loc[common,col]
            changed=delta.abs().gt(1e-10)
            row[col+"_mismatch_targets"]=len(set(common.get_level_values("target_model")[changed]))
            row[col+"_max_abs_difference"]=float(delta.abs().max())
            if changed.any():
                local=pd.DataFrame({
                    "origin_model":common.get_level_values("origin_model"),
                    "target_model":common.get_level_values("target_model"),
                    "value_a":da.loc[common,col].to_numpy(),
                    "value_b":db.loc[common,col].to_numpy(),
                    "difference":delta.to_numpy(),
                }).loc[changed.to_numpy()].copy()
                local["field"]=col;local["attack_a"]=a;local["attack_b"]=b
                detail.append(local.drop(columns="origin_model").drop_duplicates())
            if changed.any() and col in ["num_samples","target_threshold"]:
                raise ValueError(f"{a} versus {b}: {col} differs; check evaluation settings")
        rows.append(row)
    return pd.DataFrame(rows), pd.concat(detail,ignore_index=True) if detail else pd.DataFrame()


def summarize_attacks(df):
    rows=[]
    for a in ATTACK_ORDER:
        d=df[df.attack_type.eq(a)]
        if d.empty:continue
        for label,mask in [("White-box",d.is_whitebox),("Black-box",d.is_blackbox)]:
            s=d.loc[mask,"asr"]
            rows.append({"attack_type":a,"attack":ATTACK_LABELS[a],"evaluation":label,
                         "pairs":len(s),"mean_asr_pct":100*s.mean(),
                         "median_asr_pct":100*s.median(),"min_asr_pct":100*s.min(),
                         "max_asr_pct":100*s.max()})
    return pd.DataFrame(rows)


def summarize_targets(df):
    wb=df[df.is_whitebox][["attack_type","target_model","asr"]].rename(columns={"asr":"whitebox_asr"})
    bb=df[df.is_blackbox].groupby(["attack_type","target_model"],as_index=False).asr.mean().rename(
        columns={"asr":"mean_blackbox_asr"})
    return wb.merge(bb,on=["attack_type","target_model"],validate="one_to_one")


def presentation_stats(stats_df):
    out=stats_df.copy()
    for col in ["mean_true","mean_false","mean_diff","ci_low","ci_high","se_node_jackknife"]:
        if col in out:out[col+"_pp"]=100*out[col]
    return out


def write_result_tables(stats_df,diffs,jackknifes,out_dir):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    stats_df.to_csv(out_dir/"matched_statistics_all_rows.csv",index=False)
    presentation_stats(stats_df[stats_df.primary_holm_family]).to_csv(
        out_dir/"matched_statistics_three_attacks.csv",index=False)
    presentation_stats(stats_df[stats_df.legacy_holm_family]).to_csv(
        out_dir/"matched_statistics_original_19.csv",index=False)
    for folder,data in [("matched_strata",diffs),("node_jackknife",jackknifes)]:
        root=out_dir/folder;root.mkdir(exist_ok=True)
        for key,d in data.items():
            d.to_csv(root/(key+".csv"),index=False)


def _save_figure(fig,out_dir,stem):
    if out_dir is not None:
        out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
        fig.savefig(out_dir/(stem+".pdf"),bbox_inches="tight")
        fig.savefig(out_dir/(stem+".png"),dpi=240,bbox_inches="tight")


def make_asr_cmap():
    import matplotlib.colors as mcolors
    return mcolors.LinearSegmentedColormap.from_list(
        "asr_map",[(0.0,"#c7e9c0"),(0.25,"#ffd92f"),(0.55,"#ff8c00"),(1.0,"#b30000")])


def _model_order(df):
    ids=pd.concat([df.origin_model,df.target_model]).drop_duplicates()
    def key(name):
        m=parse_model_name(name)
        def rank(x,order):return order.index(x) if x in order else len(order)
        return (rank(m.pretrain,PRETRAIN_ORDER),rank(m.backbone,BACKBONE_ORDER),
                rank(m.train_family,TRAIN_FAMILY_ORDER),name)
    return sorted(ids,key=key)


def plot_transfer_heatmap(df,title,out_dir=None,stem="transfer_matrix",model_order=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ids=_model_order(df) if model_order is None else model_order
    matrix=100*df.pivot(index="target_model",columns="origin_model",values="asr").reindex(index=ids,columns=ids)
    fig,ax=plt.subplots(figsize=(19,17))
    sns.heatmap(matrix,ax=ax,cmap=make_asr_cmap(),vmin=0,vmax=100,annot=True,fmt=".0f",
                annot_kws={"fontsize":4.2},cbar_kws={"label":"ASR (%)"})
    ax.tick_params(axis="both",labelsize=6)
    ax.set(xlabel="Source detector",ylabel="Target detector",title=title)
    fig.tight_layout();_save_figure(fig,out_dir,stem)
    return fig,matrix


def plot_group_heatmaps(df,out_dir=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.patches import Rectangle
    specs=[
        ("Architecture family","arch_family",ARCH_ORDER,{"cnn":"CNN","transformer":"Transformer"},True),
        ("Backbone","backbone",BACKBONE_ORDER,BACKBONE_LABELS,False),
        ("Pretraining","pretrain",PRETRAIN_ORDER,{"imgnet":"ImageNet","fr":"FaceRec."},False),
        ("Training data","train_family",TRAIN_FAMILY_ORDER,{},False),
    ]
    attacks=[a for a in ATTACK_ORDER if a in set(df.attack_type)]
    matrices={}
    for a in attacks:
        for title,factor,order,labels,exclude in specs:
            d=df[df.attack_type.eq(a)&df.is_blackbox]
            if exclude:d=d[~d.same_backbone]
            m=100*d.pivot_table(index="target_"+factor,columns="origin_"+factor,values="asr",aggfunc="mean")
            matrices[(a,factor)]=m.reindex(index=order,columns=order)
    vmax=max(40,10*np.ceil(max(np.nanmax(m.to_numpy()) for m in matrices.values())/10))
    fig,axes=plt.subplots(len(attacks),4,figsize=(15,3.6*len(attacks)),squeeze=False,
                          gridspec_kw={"width_ratios":[1.1,2.2,1.1,1.9]},layout="constrained")
    for i,a in enumerate(attacks):
        for j,(title,factor,order,labels,exclude) in enumerate(specs):
            m=matrices[(a,factor)].rename(index=labels,columns=labels)
            sns.heatmap(m,ax=axes[i,j],cmap=make_asr_cmap(),vmin=0,vmax=vmax,
                        annot=True,fmt=".0f",cbar=False,annot_kws={"fontsize":8})
            axes[i,j].set_title(title if i==0 else "")
            axes[i,j].set_xlabel("Source")
            axes[i,j].set_ylabel(ATTACK_LABELS[a]+"\nTarget" if j==0 else "Target")
            axes[i,j].tick_params(axis="x",labelrotation=35,labelsize=8)
            axes[i,j].tick_params(axis="y",labelsize=8)
            for k in range(len(m)):
                axes[i,j].add_patch(Rectangle((k,k),1,1,fill=False,edgecolor="black",lw=1.4))
    fig.colorbar(axes[0,0].collections[0],ax=axes.ravel().tolist(),label="Mean black-box ASR (%)",shrink=.75)
    fig.suptitle("White-box pairs excluded; architecture-family panels also exclude exact backbone matches",fontsize=11)
    _save_figure(fig,out_dir,"group_transfer_factors_three_attacks")
    return fig,matrices


def plot_compatibility_effects(stats_df,out_dir=None):
    import matplotlib.pyplot as plt
    attacks=[a for a in ATTACK_ORDER if (stats_df.attack_type.eq(a)&stats_df.scope.eq("per_attack")).any()]
    colors={"aa":"#4C78A8","cw_no_eot":"#59A14F","cw_eot":"#E15759"}
    fig,ax=plt.subplots(figsize=(10,5))
    for j,a in enumerate(attacks):
        d=stats_df[stats_df.scope.eq("per_attack")&stats_df.attack_type.eq(a)].set_index("test_name").reindex(COMPATIBILITY_TESTS)
        y=np.arange(4)+(j-(len(attacks)-1)/2)*.2
        effect=100*d.mean_diff.to_numpy();lo=100*d.ci_low.to_numpy();hi=100*d.ci_high.to_numpy()
        ax.errorbar(effect,y,xerr=np.vstack([effect-lo,hi-effect]),fmt="o",capsize=3,
                    color=colors[a],label=ATTACK_LABELS[a])
    ax.axvline(0,color=".5",ls="--",lw=1)
    ax.set_yticks(range(4),[TEST_LABELS[x] for x in COMPATIBILITY_TESTS])
    ax.invert_yaxis();ax.set_xlabel("Matched difference in ASR (percentage points)")
    ax.set_title("Compatibility contrasts with pointwise 95% node-jackknife CIs")
    ax.legend(frameon=False);ax.grid(axis="x",alpha=.2)
    fig.tight_layout();_save_figure(fig,out_dir,"compatibility_effects_three_attacks")
    return fig


def eot_ablation_table(stats_df):
    mask=stats_df.attack_type.eq("cw_eot_minus_cw_no_eot")&stats_df.scope.isin(["cross_attack","attack_interaction"])
    d=stats_df[mask].copy()
    order=["attack_cw_eot_vs_cw_no_eot"]+["attack_interaction_"+x for x in COMPATIBILITY_TESTS]
    if d.empty:return presentation_stats(d)
    d=d.set_index("test_name").reindex(order).reset_index()
    d["comparison_label"]=["Overall paired black-box ASR"]+[TEST_LABELS[x]+" contrast" for x in COMPATIBILITY_TESTS]
    return presentation_stats(d)


def plot_eot_ablation(stats_df,target_summary,out_dir=None):
    import matplotlib.pyplot as plt
    d=eot_ablation_table(stats_df)
    target=target_summary.pivot(index="target_model",columns="attack_type",values="mean_blackbox_asr")
    fig,axes=plt.subplots(1,2,figsize=(13,5.2),gridspec_kw={"width_ratios":[1,1.6]})
    axes[0].scatter(100*target.cw_no_eot,100*target.cw_eot,s=28,alpha=.8)
    upper=max(10,float(np.ceil(100*target[["cw_no_eot","cw_eot"]].max().max()/10)*10))
    axes[0].plot([0,upper],[0,upper],ls="--",c=".5")
    axes[0].set(xlim=(0,upper),ylim=(0,upper),xlabel="CW without EOT: mean black-box ASR (%)",
                ylabel="CW-EOT: mean black-box ASR (%)",title="Each point is one target detector")
    effect=d.mean_diff_pp.to_numpy();lo=d.ci_low_pp.to_numpy();hi=d.ci_high_pp.to_numpy()
    axes[1].errorbar(effect,np.arange(len(d)),xerr=np.vstack([effect-lo,hi-effect]),fmt="o",capsize=3,color="#007E87")
    axes[1].set_yticks(range(len(d)),d.comparison_label)
    axes[1].invert_yaxis();axes[1].axvline(0,ls="--",c=".5")
    axes[1].set(xlabel="CW-EOT minus CW without EOT (percentage points)",
                title="Paired configuration contrasts; pointwise 95% CIs")
    fig.tight_layout();_save_figure(fig,out_dir,"cw_eot_ablation")
    return fig


def build_clean_performance_table(df):
    d=df.copy()
    def pretrain(row):
        text=f"{row.root_name} {row.root_path}".lower()
        if "fr_pretrain" in text:return "fr"
        if "no_pretrain" in text or "nopretrain" in text:return "nopretrain"
        if str(row.root_name).lower()=="saved_models":return "imgnet"
        return "unknown"
    def backbone(name):
        s=str(name).lower()
        return next((x for x in BACKBONE_ORDER if s.startswith(x)),"unknown")
    d["pretrain"]=d.apply(pretrain,axis=1)
    d["backbone"]=d.model_name.map(backbone)
    d["train_family"]=d.model_name.astype(str).str.upper().str.extract(r"_TRAIN_(ALL|EFS|FE|FR|FS)_FF_",expand=False)
    d["auc_unit"]=pd.to_numeric(d.auc,errors="coerce")
    d.loc[d.auc_unit>1.5,"auc_unit"]/=100
    d=d[d.phase.eq("test")&d.pretrain.isin(PRETRAIN_ORDER)&d.backbone.isin(BACKBONE_ORDER)
        &d.train_family.isin(TRAIN_FAMILY_ORDER)].copy()
    keys=["pretrain","backbone","train_family","split"]
    conflicting=d.groupby(keys).auc_unit.nunique().gt(1)
    if conflicting.any():
        warnings.warn(f"{int(conflicting.sum())} clean-performance groups have different duplicate AUCs. "
                      "Preserving the original notebook's maximum aggregation; inspect source eval runs.")
    lookup=d.groupby(keys).auc_unit.max().to_dict()
    def fmt(v):return "--" if not np.isfinite(v) else f"{v:.2f}"
    rows=[]
    for backbone in BACKBONE_ORDER:
        row={"Backbone":BACKBONE_LABELS[backbone]}
        for pre in PRETRAIN_ORDER:
            for train in TRAIN_FAMILY_ORDER:
                all_=lookup.get((pre,backbone,train,"ALL_ff"),np.nan)
                cdf=lookup.get((pre,backbone,train,"ALL_cdf"),np.nan)
                id_=lookup.get((pre,backbone,train,train+"_ff"),np.nan)
                row[f"{pre}_{train}"]=f"{fmt(all_)}/{fmt(cdf)}" if train=="ALL" else f"{fmt(all_)}({fmt(id_)})/{fmt(cdf)}"
        rows.append(row)
    return pd.DataFrame(rows)


# Explicit portfolio labels: legacy combined always means AA + CW-EOT.
# Give three-attack results an unambiguous aa_cw_eot_cw_no_eot or all_three prefix.
ORACLE_PORTFOLIOS = {
    "aa": ("AA", ["aa"]),
    "cw_no_eot": ("CW without EOT", ["cw_no_eot", "cw"]),
    "cw_eot": ("CW-EOT", ["cw_eot"]),
    "combined": ("AA + CW-EOT", ["combined", "aa_cw_eot"]),
    "aa_cw_no_eot": ("AA + CW without EOT", ["aa_cw_no_eot"]),
    "cw_eot_cw_no_eot": ("CW-EOT + CW without EOT", ["cw_eot_cw_no_eot"]),
    "all_three": ("All three attacks", ["aa_cw_eot_cw_no_eot", "all_three", "combined_2"]),
}


def _oracle_path(data_dir, attack, scope, suffix):
    aliases = ORACLE_PORTFOLIOS[attack][1]
    return resolve_input_file(data_dir,
        [f"transfer_results_{alias}_{scope}_{suffix}.csv" for alias in aliases],
        required=False)


def load_oracle_summaries(data_dir, model_ids):
    """Load supplied portfolios explicitly, accepting cw as the no-EOT alias."""
    model_ids = {m for m in model_ids if "clip" not in str(m).lower()}
    frames = []; coverage = []
    for attack, (label, _) in ORACLE_PORTFOLIOS.items():
        for scope in ["strict", "self_excluded"]:
            path = _oracle_path(data_dir, attack, scope, "oracle")
            if path is None:
                coverage.append(dict(attack=attack, scope=scope, status="not supplied", targets=0))
                continue
            d = pd.read_csv(path)
            required = {"target_model", "asr"}
            if required - set(d):
                raise ValueError(f"{path}: missing oracle fields")
            if "aggregation" in d and d.aggregation.astype(str).str.contains("individual",case=False).any():
                raise ValueError(f"{path}: individual results cannot be treated as an oracle")
            d = d[d.target_model.isin(model_ids)].copy()
            if d.target_model.duplicated().any():
                raise ValueError(f"{path}: duplicate oracle targets")
            if not d.asr.between(0, 1).all():
                raise ValueError(f"{path}: invalid oracle ASR")
            if {"num_success", "num_clean_correct"} <= set(d):
                expected = d.num_success / d.num_clean_correct.replace(0, np.nan)
                if not np.allclose(d.asr, expected, atol=1e-10, rtol=0, equal_nan=True):
                    raise ValueError(f"{path}: oracle ASR disagrees with counts")
            if "oracle_rule_json" in d:
                for raw_rule in d.oracle_rule_json.dropna().unique():
                    rule = json.loads(raw_rule)
                    if not rule.get("exclude_self_origin", False):
                        raise ValueError(f"{path}: oracle does not exclude self")
                    if scope == "strict" and not all(rule.get(k, False) for k in
                        ["exclude_same_training_data", "exclude_same_architecture"]):
                        raise ValueError(f"{path}: strict oracle exclusions missing")
            d["attack"] = attack; d["scope"] = scope; d["input_file"] = str(path)
            frames.append(d)
            coverage.append(dict(attack=attack, scope=scope,
                status="complete" if len(d)==len(model_ids) else "partial", targets=len(d)))
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(),
            pd.DataFrame(coverage))


def compare_oracle_portfolios(oracle_df, baseline="combined", expanded="all_three"):
    """Compare target-level ASRs; equal counts cannot verify clean-image identities."""
    rows=[]
    keys=["target_model", "scope"]
    old=oracle_df[oracle_df.attack.eq(baseline)].copy()
    new=oracle_df[oracle_df.attack.eq(expanded)].copy()
    if old.empty or new.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged=old.merge(new,on=keys,suffixes=("_baseline","_expanded"),validate="one_to_one")
    for scope, g in merged.groupby("scope"):
        gain=g.asr_expanded-g.asr_baseline
        checks={}
        for col in ["num_clean_correct", "num_samples", "target_threshold"]:
            if col+"_baseline" in g and col+"_expanded" in g:
                checks[col+"_matches"]=bool(g[col+"_baseline"].eq(g[col+"_expanded"]).all())
            else:
                checks[col+"_matches"]=None
        if checks.get("num_clean_correct_matches") is False:
            warnings.warn(f"{scope}: oracle portfolios have different clean-correct counts; interpret ASR differences cautiously.")
        rows.append(dict(scope=scope,baseline=baseline,expanded=expanded,targets=len(g),
            mean_baseline_asr_pct=100*g.asr_baseline.mean(),
            mean_expanded_asr_pct=100*g.asr_expanded.mean(),
            mean_gain_pp=100*gain.mean(),
            improved_targets=int((gain>1e-12).sum()),
            unchanged_targets=int((gain.abs()<=1e-12).sum()),
            decreased_targets=int((gain < -1e-12).sum()),**checks))
    merged["gain_pp"]=100*(merged.asr_expanded-merged.asr_baseline)
    return pd.DataFrame(rows), merged


def plot_vulnerability_summary(target_summary,oracle_df,out_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    labels={a: spec[0] for a, spec in ORACLE_PORTFOLIOS.items()}
    available = set(oracle_df.attack) if not oracle_df.empty else set()
    order=[a for a in ORACLE_PORTFOLIOS if a in ATTACK_ORDER or a in available]
    rows=[]
    for metric in ["whitebox_asr","mean_blackbox_asr"]:
        for _,r in target_summary.iterrows():
            rows.append({"attack":r.attack_type,"metric":metric,"value":100*r[metric],"target_model":r.target_model})
    if not oracle_df.empty:
        for _,r in oracle_df.iterrows():
            metric="strict_oracle" if r.scope=="strict" else "self_excluding_oracle"
            rows.append({"attack":r.attack,"metric":metric,"value":100*r.asr,"target_model":r.target_model})
    data=pd.DataFrame(rows)
    metrics=[("whitebox_asr","White-box ASR"),("mean_blackbox_asr","Mean black-box ASR"),
             ("strict_oracle","Strict oracle"),("self_excluding_oracle","Self-excluding oracle")]
    fig,axes=plt.subplots(1,4,figsize=(20,max(4, .65*len(order))),sharey=True,layout="constrained")
    rng=np.random.default_rng(42)
    for ax,(metric,title) in zip(axes,metrics):
        for i,a in enumerate(order):
            vals=data.loc[data.metric.eq(metric)&data.attack.eq(a),"value"].to_numpy()
            if vals.size:
                ax.scatter(vals,i+rng.uniform(-.10,.10,len(vals)),s=10,alpha=.45,color="#4C78A8")
                q=np.quantile(vals,[.1,.25,.5,.75,.9])
                ax.plot([q[0],q[4]],[i,i],color="black",lw=1)
                ax.plot([q[1],q[3]],[i,i],color="black",lw=4)
                ax.scatter([q[2]],[i],color="black",s=24,marker="D",zorder=4)
            else:
                msg="not applicable" if a not in ATTACK_ORDER and "oracle" not in metric else "not supplied"
                ax.text(50,i,msg,ha="center",va="center",fontsize=8,color=".5")
        ax.set(xlim=(-3,103),xticks=[0,25,50,75,100],title=title,xlabel="ASR (%)")
        ax.set_yticks(range(len(order)),[labels[a] for a in order])
        ax.grid(axis="x",alpha=.2)
    axes[0].set_ylim(len(order)-.5,-.5)
    fig.suptitle("Detector vulnerability and supplied attack-portfolio oracles",fontsize=16)
    _save_figure(fig,out_dir,"model_vulnerability_summary_three_attacks")
    return fig,data


def plot_oracle_source_counts(data_dir,model_ids,out_dir=None):
    """Selected-source matrices for every supplied portfolio and oracle rule."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    model_ids = {m for m in model_ids if "clip" not in str(m).lower()}
    tables={}; figures=[]
    for attack, (label, _) in ORACLE_PORTFOLIOS.items():
        for scope in ["strict", "self_excluded"]:
            path = _oracle_path(data_dir, attack, scope, "oracle_source_counts")
            if path is None: continue
            d=pd.read_csv(path)
            if "origin_key" in d: d["origin_model"]=d.origin_key
            elif "origin_model" in d: d["origin_model"]=canonicalize_origin_models(d)
            else: raise ValueError(f"{path}: no source identifier")
            value=next((c for c in ["count","selected_count","num_selected","selection_count","num_images"] if c in d),None)
            if value is None: raise ValueError(f"{path}: no recognized count column")
            d=d[d.target_model.isin(model_ids)&d.origin_model.isin(model_ids)]
            m=d.pivot_table(index="target_model",columns="origin_model",values=value,aggfunc="sum",fill_value=0)
            fig,ax=plt.subplots(figsize=(22,19))
            sns.heatmap(m,ax=ax,cmap="Blues",cbar_kws={"label":"Selected images"})
            ax.set_title(f"{label} {scope} oracle: selected source counts",fontsize=18)
            ax.set(xlabel="Source",ylabel="Target"); ax.tick_params(labelsize=10)
            fig.tight_layout(); key=attack+"_"+scope
            _save_figure(fig,out_dir,key+"_source_counts")
            tables[key]=m; figures.append(fig)
    return figures,tables


def audit_oracle_details(data_dir, oracle_df, model_ids):
    """Audit winners/counts and eligibility; winners alone cannot recover all portfolios."""
    rows=[]
    for (attack, scope), summary in oracle_df.groupby(["attack", "scope"]):
        path = _oracle_path(data_dir, attack, scope, "oracle_selection")
        count_path = _oracle_path(data_dir, attack, scope, "oracle_source_counts")
        row=dict(attack=attack, scope=scope, selections="not supplied", source_counts="not supplied")
        if path is not None:
            sel=pd.read_csv(path)
            sel=sel[sel.target_model.isin(model_ids)].copy()
            if sel.duplicated(["target_model", "id"]).any():
                raise ValueError(f"{path}: duplicate target/image winners")
            if not sel.selected_prob_fake.between(0,1).all():
                raise ValueError(f"{path}: invalid selected probabilities")
            sources=canonicalize_origin_models(sel)
            if sources.eq(sel.target_model).any():
                raise ValueError(f"{path}: self-source selected")
            # Parse the same metadata as the primary transfer analysis.
            metadata={name: parse_model_name(name) for name in set(sources) | set(sel.target_model)}
            annotated=pd.DataFrame(index=sel.index)
            for field in ["backbone", "train_family"]:
                lookup={name: meta[field] for name, meta in metadata.items()}
                annotated["origin_"+field]=sources.map(lookup)
                annotated["target_"+field]=sel.target_model.map(lookup)
            if scope == "strict":
                invalid=(annotated.origin_backbone.eq(annotated.target_backbone)
                    | annotated.origin_train_family.eq(annotated.target_train_family)
                    | annotated.origin_train_family.eq("ALL"))
                if invalid.any(): raise ValueError(f"{path}: strict source exclusions violated")
            check=summary.set_index("target_model")
            sizes=sel.groupby("target_model").size().reindex(check.index)
            if not sizes.eq(check.num_samples).all():
                raise ValueError(f"{path}: selected image counts differ from oracle summary")
            thresholds=sel.target_model.map(check.target_threshold)
            if not ((sel.selected_prob_fake >= thresholds).astype(int)==sel.selected_pred).all():
                raise ValueError(f"{path}: selected labels do not match thresholds")
            cand=sel.groupby("target_model").num_candidates_for_id.agg(["mean","min","max"]).reindex(check.index)
            for statistic in ["mean", "min", "max"]:
                if not np.allclose(cand[statistic],check[statistic+"_candidates_per_image"]):
                    raise ValueError(f"{path}: candidate counts differ from oracle summary")
            row.update(selections="passed", selection_rows=len(sel))
            if count_path is not None:
                counts=pd.read_csv(count_path)
                counts=counts[counts.target_model.isin(model_ids)]
                keys=["target_model", "origin_key"] + [
                    col for col in ["attack_name", "attack_path"]
                    if col in counts.columns and col in sel.columns
                ]
                observed=sel.groupby(keys).size().rename("observed")
                reported=counts.groupby(keys).num_selected.sum().rename("reported")
                both=pd.concat([observed,reported],axis=1).fillna(0)
                if not both.observed.eq(both.reported).all():
                    raise ValueError(f"{count_path}: source counts differ from selected winners")
                row["source_counts"]="passed"
        rows.append(row)
    return pd.DataFrame(rows)


def run_group_sensitivity(df,alpha=0.05,with_cis=True,progress=True):
    frames=[]
    for factor,origin_col,target_col in [
        ("backbone","origin_backbone","target_backbone"),
        ("training_family","origin_train_family","target_train_family")]:
        levels=sorted(set(df[origin_col])|set(df[target_col]))
        for level in levels:
            if progress:print(f"Leaving out {factor}: {level}",flush=True)
            d=df[df[origin_col].ne(level)&df[target_col].ne(level)].copy()
            s,_,_=run_all_matched_tests(d,alpha=alpha,compute_inference=with_cis)
            s=s[s.primary_holm_family].copy()
            s["omission_factor"]=factor;s["omitted_level"]=level
            s["num_rows_remaining"]=len(d)
            frames.append(s)
    return pd.concat(frames,ignore_index=True)


def summarize_group_sensitivity(full,loo):
    keys=["scope","attack_type","test_name"]
    base=full[full.primary_holm_family][keys+["mean_diff","ci_low","ci_high"]].rename(
        columns={"mean_diff":"full_effect","ci_low":"full_ci_low","ci_high":"full_ci_high"})
    d=loo.merge(base,on=keys,validate="many_to_one")
    d["change_from_full"]=d.mean_diff-d.full_effect
    d["abs_change_from_full"]=d.change_from_full.abs()
    d["sign_preserved"]=np.sign(d.mean_diff)==np.sign(d.full_effect)
    records=[]
    for key,group in d.groupby(keys,sort=False):
        valid=group.dropna(subset=["mean_diff"])
        if valid.empty:continue
        worst=valid.loc[valid.abs_change_from_full.idxmax()]
        row=dict(zip(keys,key))
        row.update(full_effect_pp=100*worst.full_effect,full_ci_low_pp=100*worst.full_ci_low,
                   full_ci_high_pp=100*worst.full_ci_high,
                   maximum_abs_change_pp=100*worst.abs_change_from_full,
                   worst_omission_factor=worst.omission_factor,worst_omitted_level=worst.omitted_level,
                   sign_always_preserved=bool(valid.sign_preserved.all()),
                   nonestimable_refits=int(group.mean_diff.isna().sum()))
        for factor in ["backbone","training_family"]:
            g=valid[valid.omission_factor.eq(factor)]
            row[f"{factor}_out_min_pp"]=100*g.mean_diff.min()
            row[f"{factor}_out_max_pp"]=100*g.mean_diff.max()
        records.append(row)
    return d,pd.DataFrame(records)


def plot_group_sensitivity(full,loo,out_dir=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    panels=[
        ("Compatibility contrasts",full.scope.eq("per_attack")&full.test_name.isin(COMPATIBILITY_TESTS)),
        ("Between-attack compatibility contrasts",full.scope.eq("attack_interaction")),
        ("Overall paired attack contrasts",full.scope.eq("cross_attack")),
    ]
    counts=[max(1,int(mask.sum())) for _,mask in panels]
    fig,axes=plt.subplots(3,1,figsize=(13,15),gridspec_kw={"height_ratios":counts})
    keys=["scope","attack_type","test_name"]
    for ax,(title,mask) in zip(axes,panels):
        rows=full[mask].copy().reset_index(drop=True)
        labels=[]
        for i,row in rows.iterrows():
            if row.scope=="per_attack":
                label=TEST_LABELS.get(row.test_name,row.test_name)+" ["+ATTACK_LABELS[row.attack_type]+"]"
            elif row.scope=="attack_interaction":
                name=row.test_name.replace("attack_interaction_","")
                a,b=row.attack_type.split("_minus_")
                label=TEST_LABELS.get(name,name)+" ["+ATTACK_LABELS[a]+" - "+ATTACK_LABELS[b]+"]"
            else:
                a,b=row.attack_type.split("_minus_")
                label=ATTACK_LABELS[a]+" - "+ATTACK_LABELS[b]
            labels.append(label)
            sub=loo.copy()
            for key in keys:sub=sub[sub[key].eq(row[key])]
            for factor,color,offset in [("backbone","#4C78A8",-.14),("training_family","#F28E2B",.14)]:
                g=sub[sub.omission_factor.eq(factor)].dropna(subset=["mean_diff"])
                y=i+offset+np.linspace(-.065,.065,len(g))
                x=100*g.mean_diff.to_numpy()
                lo=100*g.ci_low.to_numpy();hi=100*g.ci_high.to_numpy()
                finite=np.isfinite(lo)&np.isfinite(hi)
                if finite.any():
                    ax.errorbar(x[finite],y[finite],xerr=np.vstack([x[finite]-lo[finite],hi[finite]-x[finite]]),
                                fmt="none",ecolor=color,alpha=.3,lw=.8)
                ax.scatter(x,y,s=13,color=color,alpha=.75)
            effect=100*row.mean_diff;low=100*row.ci_low;high=100*row.ci_high
            ax.errorbar([effect],[i],xerr=[[effect-low],[high-effect]],fmt="D",color="black",ms=4,capsize=2)
        ax.set_yticks(range(len(rows)),labels,fontsize=8)
        ax.invert_yaxis();ax.axvline(0,ls="--",c=".5",lw=.8)
        ax.set_title(title,loc="left",fontsize=11)
        ax.set_xlabel("Difference in ASR (percentage points)");ax.grid(axis="x",alpha=.2)
    axes[0].legend(handles=[
        Line2D([0],[0],marker="D",color="black",lw=1,label="Full estimate and pointwise 95% CI"),
        Line2D([0],[0],marker="o",color="#4C78A8",lw=0,label="Backbone omission"),
        Line2D([0],[0],marker="o",color="#F28E2B",lw=0,label="Training-data omission"),
    ],fontsize=8,frameon=False)
    fig.tight_layout();_save_figure(fig,out_dir,"group_omission_sensitivity_three_attacks")
    return fig
