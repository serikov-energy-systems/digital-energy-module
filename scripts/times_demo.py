"""Run the data-centre processes inside the open two-region TIMES demo, rigid and with time slice storage.

    TIMES_DEMO=<clone of etsap-TIMES/TIMES_Demo> GAMS=<gams executable> python scripts/times_demo.py

Needs GAMS (the demo licence is enough: the demo is reduced to the milestone years 2005 and 2030) and the Python
package gamsapi to read the solution. Writes results/times/*.dd and results/times_demo.csv.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import gams.transfer as gt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from digienergy import digital, times_link  # noqa: E402

DEMO = Path(os.environ.get("TIMES_DEMO", ROOT.parent / "times-demo"))
GAMS = os.environ.get("GAMS", "gams")
SITES = {"REG1": "DE", "REG2": "SE"}            # temperature series used for the two hypothetical demo regions
DEMAND_SHARE = 0.03                              # data centres: share of the region's electricity output in 2030
YEAR = 2011


def run_file(extra: str | None) -> str:
    text = (DEMO / "model" / "demo12.run").read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"(?im)^\s*option\s+LP\s*=\s*\w+\s*;", "option LP=CPLEX;", text)
    text = re.sub(r"(?im)^SET MILESTONYR\s*/.*/;", "SET MILESTONYR /2005,2030/;", text)
    if extra:
        text = text.replace("$BATINCLUDE bounds-uc_wsets.dd", f"$BATINCLUDE bounds-uc_wsets.dd\n$BATINCLUDE {extra}")
    return text


def solve(work: Path, name: str, extra: str | None) -> gt.Container:
    (work / f"{name}.run").write_text(run_file(extra), encoding="utf-8")
    subprocess.run([GAMS, f"{name}.run", f"idir={DEMO / 'source'}", f"gdx={name}", "lo=0"], cwd=work, check=True)
    listing = (work / f"{name}.lst").read_text(encoding="utf-8", errors="replace")
    status = re.findall(r"MODEL STATUS\s+(\d+)", listing)
    if not status or status[-1] != "1":
        raise RuntimeError(f"{name}: model status {status[-1:]}")
    return gt.Container(str(work / f"{name}.gdx"))


def records(c: gt.Container, name: str) -> pd.DataFrame:
    d = c[name].records.copy()
    for column in d.columns[:-1]:
        d[column] = d[column].astype(str)
    return d


def main() -> None:
    a = digital.Assumptions()
    results = ROOT / "results" / "times"
    results.mkdir(parents=True, exist_ok=True)
    temperature = pd.read_csv(ROOT / "data" / f"temperature_{YEAR}.csv", index_col=0, parse_dates=True)
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        for f in (DEMO / "model").iterdir():
            if f.suffix in (".dd", ".opt"):
                shutil.copy(f, work / f.name)
        base = solve(work, "base", None)
        produced = records(base, "PAR_COMPRDL")
        produced.columns = ["r", "year", "c", "s", "value"]
        electricity = produced[(produced.c == "ELC") & (produced.year == "2030")].groupby("r").value.sum()
        settings = (DEMO / "model" / "syssettings.dd").read_text(encoding="utf-8", errors="replace")
        fractions = {r: {s: float(re.search(rf"{r}\.{s}\s+([0-9.]+)", settings).group(1)) for s in times_link.SLICES}
                     for r in SITES}
        tables = {r: times_link.slice_table(temperature[z], fractions[r], a) for r, z in SITES.items()}
        demand = {r: float(DEMAND_SHARE * electricity[r]) for r in SITES}
        rows = [{"case": "no data centres", "objective": float(base["OBJZ"].records.level.iloc[0])}]
        for case, flexible in (("data centres, rigid", False), ("data centres, batch storage", True)):
            name = "dc_flex" if flexible else "dc_rigid"
            text = times_link.write_dd(tables, demand, a, flexible)
            (work / f"{name}.dd").write_text(text, encoding="utf-8")
            (results / f"{name}.dd").write_text(text, encoding="utf-8", newline="\n")
            c = solve(work, name, f"{name}.dd")
            flows = records(c, "PAR_FLO")
            flows.columns = ["r", "vintage", "year", "p", "c", "s", "value"]
            now = flows[(flows.year == "2030") & flows.p.isin(["DCINT", "DCBAT"])]
            price = records(c, "EQE_COMBAL") if "EQE_COMBAL" in c.data else None
            row = {"case": case, "objective": float(c["OBJZ"].records.level.iloc[0])}
            for r in SITES:
                use = now[(now.r == r) & (now.c == "ELC")].groupby("s").value.sum()
                work_done = now[(now.r == r) & (now.p == "DCBAT") & (now.c == "ITB")].groupby("s").value.sum()
                out = now[(now.r == r) & (now.c.isin(["ITSINT", "ITB"]))].groupby("s").value.sum()
                row[f"{r}_it_demand_pj"] = demand[r]
                row[f"{r}_electricity_pj"] = float(use.sum())
                row[f"{r}_mean_pue"] = float(use.sum() / out.sum())
                for s in times_link.SLICES:
                    row[f"{r}_batch_{s}_pj"] = float(work_done.get(s, 0.0))
                if price is not None:
                    q = price.iloc[:, :6]
                    q.columns = ["r", "year", "c", "s", "level", "marginal"]
                    q = q[(q.r == r) & (q.c == "ELC") & (q.year == "2030")]
                    for s, v in zip(q.s, q.marginal):
                        row[f"{r}_elc_price_{s}"] = float(v)
            rows.append(row)
        for r, t in tables.items():
            t.to_csv(results / f"slices_{r}.csv", float_format="%.6g")
    table = pd.DataFrame(rows)
    table["added_objective"] = table.objective - table.objective.iloc[0]
    table.to_csv(ROOT / "results" / "times_demo.csv", index=False, float_format="%.9g")
    print(table.T.to_string())


if __name__ == "__main__":
    main()
