import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ---- CONFIG ----
csv_path = "attack_manifest_2000.csv"
out_root = Path("datasets/DF40_ATTACK_FAKE_2000")
clean_dir = out_root / "clean"
out_csv_path = out_root / "attack_manifest_2000_clean.csv"

# make output dirs
clean_dir.mkdir(parents=True, exist_ok=True)
out_root.mkdir(parents=True, exist_ok=True)

# read manifest
df = pd.read_csv(csv_path)

clean_paths = []
missing = []
copied = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    src = Path(row["frame_path"])
    sample_id = row["sample_id"]

    dst = clean_dir / f"{sample_id}.png"

    if not src.exists():
        missing.append(str(src))
        clean_paths.append(None)
        continue

    if not dst.exists():
        shutil.copy2(src, dst)
        copied += 1

    # store path relative to dataset root
    clean_paths.append(str(dst.relative_to(out_root)))

df["clean_rel_path"] = clean_paths
df.to_csv(out_csv_path, index=False)

print("\nDone.")
print(f"Output root: {out_root}")
print(f"Copied images: {copied}")
print(f"Missing images: {len(missing)}")

if missing:
    print("\nMissing paths (first 10):")
    for m in missing[:10]:
        print(m)