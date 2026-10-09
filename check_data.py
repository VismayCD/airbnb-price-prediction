"""
Run BEFORE the experiments to see exactly which paper features your CSVs provide.
 
    python check_data.py data/nyc.csv data/berlin.csv data/paris.csv
 
For every feature group it lists present columns (with % missing) and missing columns.
A column counts as "usable" when more than MIN_FILLED of its values are present.
"""
import os
import sys
 
import pandas as pd
 
from airbnb_price import (load_city, CONTINUOUS, CATEGORICAL, LIST_COLS, TEXT_COLS,
                          DATE_COLS, PRICE_CAP)
 
MIN_FILLED = 0.05  # a column needs > 5% non-missing values to count as usable
 
GROUPS = {
    "continuous (C)": CONTINUOUS,
    "categorical (O)": CATEGORICAL + LIST_COLS,
    "text (T)": TEXT_COLS,
    "date (D)": DATE_COLS,
}
 
 
def report(path: str) -> None:
    """Print the feature-coverage report for one listings CSV."""
    raw_rows = len(pd.read_csv(path, low_memory=False))
    clean = load_city(path)  # same cleaning the experiments use
 
    print(f"\n=== {path} ===")
    print(f"raw rows: {raw_rows} | usable rows (price valid and <= ${PRICE_CAP}): {len(clean)}")
 
    for group, cols in GROUPS.items():
        present = [c for c in cols if c in clean.columns]
        missing = [c for c in cols if c not in clean.columns]
        usable = [c for c in present if clean[c].notna().mean() > MIN_FILLED]
 
        print(f"\n{group}: {len(usable)}/{len(cols)} usable")
        for c in present:
            flag = "" if c in usable else "  <- mostly empty"
            print(f"   {c:35s} {clean[c].isna().mean():6.1%} missing{flag}")
        if missing:
            print("   NOT IN FILE:", ", ".join(missing))
 
 
def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("Usage: python check_data.py data/nyc.csv [data/paris.csv ...]")
    for path in paths:
        if not os.path.isfile(path):
            print(f"\n=== {path} ===\n   file not found, skipping")
            continue
        report(path)
 
 
if __name__ == "__main__":
    main()