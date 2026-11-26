"""Sensitivity to the number of representative weeks: scenarios of the first weather year with 26 weeks.

    python scripts/check_weeks.py [weeks] [scenario]

Without a scenario all key scenarios are solved; with one, only it (one process per run keeps memory low). The table
compares whatever has been solved with the 13-week results.
"""

import os
import sys
from pathlib import Path

weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 26
os.environ["DIGI_WEEKS"] = str(weeks)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from digienergy import run  # noqa: E402

main_results = run.RES
run.RES = main_results / f"weeks{weeks}"
data = run.setting()
year = run.YEARS[0]
wanted = [sys.argv[2]] if len(sys.argv) > 2 else list(run.KEY)
for name in wanted:
    run.run_one(year, name, data)
rows, base = [], {}
for name in run.KEY:
    stem = f"{year}_{name.replace(' ', '_').replace(',', '')}.json"
    if not (run.RES / "runs" / stem).exists():
        continue
    new = pd.read_json(run.RES / "runs" / stem, typ="series")
    old = pd.read_json(main_results / "runs" / stem, typ="series")
    if not base:                                  # the first key scenario is the one without new demand
        base = {"new": new["cost_bn"], "old": old["cost_bn"]}
    rows.append({"scenario": name, "cost_bn_13": old["cost_bn"], f"cost_bn_{weeks}": new["cost_bn"],
                 "added_bn_13": old["cost_bn"] - base["old"], f"added_bn_{weeks}": new["cost_bn"] - base["new"]})
table = pd.DataFrame(rows)
table.to_csv(main_results / "check_weeks.csv", index=False, float_format="%.5g")
print(table.to_string(index=False))
