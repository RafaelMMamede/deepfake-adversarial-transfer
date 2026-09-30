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


def run_all_matched_tests(df, alpha=0.05):
    df_bb = add_reverse_pair_columns(
        df[df["is_blackbox"] & df["asr"].notna()].copy()
    )

    test_specs = [
        {
            "name": "same_pretrain",
            "effect_col": "same_pretrain",
            "strata_cols": [
                "attack_type",
                "origin_backbone",
                "target_backbone",
                "origin_train_family",
                "target_train_family",
            ],
            "df": df_bb,
        },
        {
            "name": "same_backbone",
            "effect_col": "same_backbone",
            "strata_cols": [
                "attack_type",
                "origin_pretrain",
                "target_pretrain",
                "origin_train_family",
                "target_train_family",
            ],
            "df": df_bb,
        },
        {
            "name": "same_train_family_excl_ALL",
            "effect_col": "same_train_family",
            "strata_cols": [
                "attack_type",
                "origin_pretrain",
                "target_pretrain",
                "origin_backbone",
                "target_backbone",
            ],
            "df": df_bb[
                (~df_bb["origin_train_family"].eq("ALL")) &
                (~df_bb["target_train_family"].eq("ALL"))
            ].copy(),
        },
        {
            "name": "same_arch_family_excl_same_backbone",
            "effect_col": "same_arch_family",
            "strata_cols": ["attack_type", "origin_pretrain", "target_pretrain",
                            "origin_train_family", "target_train_family"],
            "df": df_bb[~df_bb["same_backbone"]].copy(),
        },
    ]

    summary_rows = []
    diffs_dict = {}
    jackknife_replicates = {}

    # Overall tests
    for spec in test_specs:
        result, diffs, replicates = matched_effect_test(
            df=spec["df"],
            effect_col=spec["effect_col"],
            strata_cols=spec["strata_cols"],
            alpha=alpha,
        )

        result["scope"] = "all_attacks"
        result["attack_type"] = "all"
        result["test_name"] = spec["name"]

        summary_rows.append(result)
        key = f"all_attacks__{spec['name']}"
        diffs_dict[key] = diffs
        jackknife_replicates[key] = replicates

    # Per-attack tests
    for attack_type, dfa in df_bb.groupby("attack_type"):
        for spec in test_specs:
            local_df = spec["df"]
            local_df = local_df[local_df["attack_type"] == attack_type].copy()

            local_strata = [
                c for c in spec["strata_cols"]
                if c != "attack_type"
            ]

            result, diffs, replicates = matched_effect_test(
                df=local_df,
                effect_col=spec["effect_col"],
                strata_cols=local_strata,
                alpha=alpha,
            )

            result["scope"] = "per_attack"
            result["attack_type"] = attack_type
            result["test_name"] = spec["name"]

            summary_rows.append(result)
            key = f"{attack_type}__{spec['name']}"
            diffs_dict[key] = diffs
            jackknife_replicates[key] = replicates

    # Direct attack-by-compatibility interactions. These are not inferred by
    # comparing two separately significant per-attack effects: each test
    # directly estimates Delta(CW-EOT) - Delta(AA), and the complete contrast
    # is recomputed inside every leave-one-node replicate.
    required_attacks = {"aa", "cw_eot"}
    observed_attacks = set(df_bb["attack_type"].dropna().astype(str))
    if required_attacks.issubset(observed_attacks):
        for spec in test_specs:
            interaction_strata = [
                c for c in spec["strata_cols"]
                if c != "attack_type"
            ]
            result, diffs, replicates = (
                attack_compatibility_interaction_test(
                    df=spec["df"],
                    effect_col=spec["effect_col"],
                    strata_cols=interaction_strata,
                    attack_a="cw_eot",
                    attack_b="aa",
                    alpha=alpha,
                )
            )

            interaction_name = f"attack_interaction_{spec['name']}"
            result.update({
                "scope": "attack_interaction",
                "attack_type": "cw_eot_minus_aa",
                "test_name": interaction_name,
            })

            summary_rows.append(result)
            diffs_dict[interaction_name] = diffs
            jackknife_replicates[interaction_name] = replicates
    else:
        print(
            "WARNING: attack-by-compatibility interactions were skipped "
            "because both 'aa' and 'cw_eot' were not present."
        )

    # Labelled tests. All contrasts are reported as first level minus second.
    labelled_specs = [
        # Exact source-target pairing across attacks.
        ("attack_cw_eot_vs_aa", "attack_type", "cw_eot", "aa",
         ["origin_model", "target_model"], df_bb, "cross_attack"),
        # Exact reverse detector pairs: A->B and B->A occur in the same model_pair.
        ("direction_transformer_to_cnn_vs_reverse", "arch_direction",
         "transformer->cnn", "cnn->transformer",
         ["attack_type", "model_pair"], df_bb, "per_attack"),
        ("direction_fr_to_imgnet_vs_reverse", "pretrain_direction",
         "fr->imgnet", "imgnet->fr",
         ["attack_type", "model_pair"], df_bb, "per_attack"),
        ("direction_all_to_single_vs_reverse", "training_direction",
         "ALL->single", "single->ALL",
         ["attack_type", "model_pair"], df_bb, "per_attack"),
    ]

    for name, label, a, b, strata, data, scope in labelled_specs:
        result, diffs, replicates = labelled_contrast_test(
            data, label, a, b, strata, name, alpha=alpha,
        )
        result.update({"scope": scope, "attack_type": "all", "test_name": name})
        summary_rows.append(result)
        diffs_dict[name] = diffs
        jackknife_replicates[name] = replicates

    # Also emit per-attack directional results (the joint versions above retain
    # attack_type as a matching stratum and summarize both attacks).
    for attack, da in df_bb.groupby("attack_type"):
        for name, label, a, b, strata, data, scope in labelled_specs[1:]:
            local = data[data["attack_type"].eq(attack)]
            local_strata = [c for c in strata if c != "attack_type"]
            result, diffs, replicates = labelled_contrast_test(
                local, label, a, b, local_strata, name,
                alpha=alpha,
            )
            result.update({"scope": "per_attack", "attack_type": attack,
                           "test_name": name})
            summary_rows.append(result)
            key = f"{attack}__{name}"
            diffs_dict[key] = diffs
            jackknife_replicates[key] = replicates

    stats_summary = pd.DataFrame(summary_rows)

    ordered_cols = [
        "scope",
        "attack_type",
        "test_name",
        "effect",
        "n_strata",
        "n_strata_attack_a",
        "n_strata_attack_b",
        "mean_true",
        "mean_false",
        "mean_diff",
        "num_nodes",
        "se_node_jackknife",
        "variance_node_jackknife",
        "jackknife_variance_multiplier",
        "reference_distribution",
        "z_node_jackknife",
        "ci_low",
        "ci_high",
        "p_node_jackknife",
        "loo_mean",
        "loo_min",
        "loo_max",
        "max_abs_loo_change",
        "most_influential_node",
    ]

    # Interaction-only metadata columns are absent in a single-attack run.
    # Create any missing output columns so the summary schema remains stable.
    for col in ordered_cols:
        if col not in stats_summary.columns:
            stats_summary[col] = np.nan

    stats_summary = stats_summary[ordered_cols]

    # Holm family-wise error correction across the 19 reported comparisons.
    # The pooled all-attacks compatibility/direction rows are descriptive
    # duplicates and are not part of this correction family.
    stats_summary = add_holm_correction(stats_summary, alpha=alpha)

    return stats_summary, diffs_dict, jackknife_replicates


def add_holm_correction(stats_summary, alpha=0.05):
    """Holm-adjust the 19 normal-approximation node-jackknife Wald p-values."""
    out = stats_summary.copy()

    per_attack_tests = {
        "same_pretrain",
        "same_backbone",
        "same_train_family_excl_ALL",
        "same_arch_family_excl_same_backbone",
        "direction_transformer_to_cnn_vs_reverse",
        "direction_fr_to_imgnet_vs_reverse",
        "direction_all_to_single_vs_reverse",
    }
    cross_attack_tests = {
        "attack_cw_eot_vs_aa",
    }
    interaction_tests = {
        "attack_interaction_same_pretrain",
        "attack_interaction_same_backbone",
        "attack_interaction_same_train_family_excl_ALL",
        "attack_interaction_same_arch_family_excl_same_backbone",
    }
    expected_family_size = (
        2 * len(per_attack_tests)
        + len(cross_attack_tests)
        + len(interaction_tests)
    )
    primary_mask = (
        (
            out["scope"].eq("per_attack")
            & out["attack_type"].isin(["aa", "cw_eot"])
            & out["test_name"].isin(per_attack_tests)
        )
        | (
            out["scope"].eq("cross_attack")
            & out["test_name"].isin(cross_attack_tests)
        )
        | (
            out["scope"].eq("attack_interaction")
            & out["attack_type"].eq("cw_eot_minus_aa")
            & out["test_name"].isin(interaction_tests)
        )
    )

    # Only finite p-values can enter the procedure. With a complete AA +
    # CW-EOT analysis this selects exactly 19 tests.
    valid_mask = primary_mask & pd.to_numeric(
        out["p_node_jackknife"], errors="coerce"
    ).notna()
    selected = out.index[valid_mask]

    out["primary_holm_family"] = primary_mask
    out["p_holm"] = np.nan
    out["reject_holm"] = False
    out["holm_family_size"] = len(selected)

    if len(selected) == 0:
        return out

    if len(selected) != expected_family_size:
        print(
            f"WARNING: Holm correction expected {expected_family_size} "
            "reported tests, "
            f"but found {len(selected)} finite p-values. Correction was "
            "applied to the available reported tests."
        )

    raw_p = out.loc[selected, "p_node_jackknife"].astype(float).to_numpy()
    order = np.argsort(raw_p)
    ordered_p = raw_p[order]
    m = len(ordered_p)

    # Holm adjusted p_(i) = max_{j<=i}[(m-j+1) p_(j)], capped at 1.
    adjusted_ordered = np.maximum.accumulate(
        (m - np.arange(m)) * ordered_p
    )
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered

    out.loc[selected, "p_holm"] = adjusted
    out.loc[selected, "reject_holm"] = adjusted <= alpha

    return out


# ============================================================
# 7. Saving / formatting
# ============================================================

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
        "Import this module, label the AA and CW-EOT frames row-wise, "
        "concatenate them, and call run_transfer_analysis(...)."
    )