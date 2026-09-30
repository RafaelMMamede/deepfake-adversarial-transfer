import json
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# config
# --------------------------------------------------
ROOTS = {
    "saved_models": Path("./saved_models"),
    # add more roots here
    "no_pretrain": Path("./saved_models_no_pretrain"),
    "fr_pretrain": Path("./saved_models_fr_pretrain"),

}

OUTPUT_CSV = "all_model_results.csv"
OUTPUT_XLSX = "all_model_results.xlsx"

# Optional: restrict to result JSON names
#JSON_PATTERN = "*.json"
JSON_PATTERN = "eval_results.json"
# JSON_PATTERN = "best_metrics*.json"


# --------------------------------------------------
# helper to infer model/run name
# --------------------------------------------------
def infer_model_name(data, json_path=None):
    checkpoint = data.get("checkpoint")

    # Prefer checkpoint path if available
    if checkpoint:
        parts = Path(checkpoint).parts

        if len(parts) >= 4:
            return parts[-4]

        return Path(checkpoint).stem

    # Fallback: use parent folder of the json
    if json_path is not None:
        return Path(json_path).parent.name

    return None


# --------------------------------------------------
# helper to flatten one results json
# --------------------------------------------------
def flatten_result_json(data, json_path=None, root_name=None, root_path=None):
    rows = []

    checkpoint = data.get("checkpoint")
    detector_config = data.get("detector_config")
    eval_config = data.get("eval_config")
    threshold_metric = data.get("threshold_metric")

    model_name = infer_model_name(data, json_path=json_path)

    base_info = {
        "root_name": root_name,
        "root_path": str(root_path) if root_path else None,
        "model_name": model_name,
        "json_path": str(json_path) if json_path else None,
        "checkpoint": checkpoint,
        "detector_config": detector_config,
        "eval_config": eval_config,
        "threshold_metric": threshold_metric,
    }

    # validation row
    if "val" in data and isinstance(data["val"], dict):
        row = {
            **base_info,
            "phase": "val",
            "split": "val",
            **data["val"],
        }
        rows.append(row)

    # test rows
    if "test" in data and isinstance(data["test"], dict):
        for split_name, metrics in data["test"].items():
            if not isinstance(metrics, dict):
                continue

            row = {
                **base_info,
                "phase": "test",
                "split": split_name,
                **metrics,
            }
            rows.append(row)

    return rows


# --------------------------------------------------
# find and parse all result jsons from multiple roots
# --------------------------------------------------
all_rows = []

for root_name, root_path in ROOTS.items():
    root_path = Path(root_path)

    if not root_path.exists():
        print(f"Skipping missing root: {root_name} -> {root_path}")
        continue

    json_files = sorted(root_path.rglob(JSON_PATTERN))

    print(f"{root_name}: found {len(json_files)} json files")

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # skip jsons that do not match this structure
            if not isinstance(data, dict):
                continue

            if "val" not in data and "test" not in data:
                continue

            all_rows.extend(
                flatten_result_json(
                    data,
                    json_path=json_file,
                    root_name=root_name,
                    root_path=root_path,
                )
            )

        except Exception as e:
            print(f"Skipping {json_file}: {e}")


df = pd.DataFrame(all_rows)

# --------------------------------------------------
# optional: reorder columns
# --------------------------------------------------
preferred_cols = [
    "root_name",
    "model_name",
    "phase",
    "split",

    "threshold",
    "auc",
    "acc",
    "precision",
    "recall",
    "f1",
    "balanced_acc",

    "tp",
    "tn",
    "fp",
    "fn",

    "threshold_metric",
    "checkpoint",
    "detector_config",
    "eval_config",
    "root_path",
    "json_path",
]

existing_preferred = [c for c in preferred_cols if c in df.columns]
remaining_cols = [c for c in df.columns if c not in existing_preferred]

if len(df) > 0:
    df = df[existing_preferred + remaining_cols]

# --------------------------------------------------
# save
# --------------------------------------------------
df.to_csv(OUTPUT_CSV, index=False)



print(df.head())
print(f"\nSaved to {OUTPUT_CSV}")
