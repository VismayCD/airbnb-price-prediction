"""
Run before the experiments to see exactly which paper features the CSVs provide.
    python check_data.py data/nyc.csv data/berlin.csv data/paris.csv
For every feature group it lists present columns (with % missing) and missing columns.
"""
import sys
import pandas as pd
from airbnb_price import (load_city, CONTINUOUS, CATEGORICAL, LIST_COLS, TEXT_COLS,
                          DATE_COLS, PRICE_CAP)

GROUPS = {"continuous (C)": CONTINUOUS, "categorical (O)": CATEGORICAL + LIST_COLS,
          "text (T)": TEXT_COLS, "date (D)": DATE_COLS}

for path in sys.argv[1:]:
    raw = pd.read_csv(path, low_memory=False)
    clean = load_city(path)                       # same cleaning the experiments use
    print(f"\n=== {path} ===")
    print(f"raw rows: {len(raw)} | usable rows (price valid and <= ${PRICE_CAP}): {len(clean)}")
    for group, cols in GROUPS.items():
        present = [c for c in cols if c in clean.columns]
        missing = [c for c in cols if c not in clean.columns]
        usable = [c for c in present if clean[c].notna().mean() > 0.05]   # >5% filled
        print(f"\n{group}: {len(usable)}/{len(cols)} usable")
        for c in present:
            print(f"   {c:35s} {clean[c].isna().mean():6.1%} missing")
        if missing:
            print("   NOT IN FILE:", ", ".join(missing))
