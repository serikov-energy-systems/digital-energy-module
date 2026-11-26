"""Scenarios of the digitalisation module on the northwestern European model, for several weather years.

    python -m digienergy.run [year ...]          one capacity expansion per scenario and year; results are cached
    python -m digienergy.run report

Inputs as in github.com/serikov-energy-systems/eu-weather-years-pypsa; their folder is taken from DIGI_DATA,
by default ../eu-weather-years-pypsa/data/raw.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib
import pandas as pd

from . import base, digital, inputs, solve, temperature

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RES, FIG = ROOT / "results", ROOT / "figures"
DATA = Path(os.environ.get("DIGI_DATA", ROOT.parent / "eu-weather-years-pypsa" / "data" / "raw"))
STEP = base.STEP_HOURS
YEARS = (2011, 1985, 1990)             # median, costliest and cheapest weather year of the base study
# all scenarios on the median year; these on the other two
KEY = ("no new digital demand", "data centres, inflexible", "batch may wait 24 h",
       "24 h and movable between countries", "24 h, sited by the model")

A = digital.Assumptions()
SCENARIOS = {  # name: (data-centre keywords or None, other changes)
    "no new digital demand": (None, {}),
    "data centres, inflexible": ({}, {}),
    "batch may wait 6 h": ({"window_h": 6}, {}),
    "batch may wait 24 h": ({"window_h": 24}, {}),
    "24 h and movable between countries": ({"window_h": 24, "pooled": True}, {}),
    "24 h, sited by the model": ({"window_h": 24, "sited": True}, {}),
    "24 h, sited, waste heat reused": ({"window_h": 24, "sited": True, "heat_reuse": True}, {}),
    "24 h, on-site batteries of 1 h": ({"window_h": 24, "battery_h": 1.0}, {}),
    "low demand (70 TWh), 24 h": ({"window_h": 24}, {"demand_twh": 70.0}),
    "high demand (220 TWh), 24 h": ({"window_h": 24}, {"demand_twh": 220.0}),
    "digital control of 2% of demand": (None, {"control": 0.02}),
    "digital control of 5% of demand": (None, {"control": 0.05}),
    "more work at home": (None, {"home": (-0.010, 0.006)}),
    "more work at home, less saving": (None, {"home": (-0.005, 0.008)}),
}


def setting():
    inputs.RAW = DATA
    cf = pd.read_parquet(DATA / "ninja.parquet") if (DATA / "ninja.parquet").exists() else inputs.load_profiles()
    demand = inputs.load_demand()
    costs = inputs.load_costs(DATA / "costs_2030.csv")
    return cf, demand, costs, inputs.transfer_limits(), inputs.existing_fleet()


def weather(cf: pd.DataFrame, year: int) -> pd.DataFrame:
    return base.representative(base.weather_year(cf, year))


def temperature_on(year: int, index: pd.DatetimeIndex) -> pd.DataFrame:
    t = temperature.load_year(year, ROOT / "data")
    t = inputs.no_leap(inputs.no_leap(t).resample(f"{STEP}h").mean().dropna(how="all"))
    return base.representative(t).iloc[: len(index)].set_axis(index)


def run_one(year: int, name: str, data) -> dict:
    path = RES / "runs" / f"{year}_{name.replace(' ', '_').replace(',', '').replace('(', '').replace(')', '').replace('%', 'pct')}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    cf, demand, costs, limits, existing = data
    dc, other = SCENARIOS[name]
    a = replace(A, demand_twh=other.get("demand_twh", A.demand_twh))
    t0 = time.time()
    n = base.build(weather(cf, year), demand, costs, limits, existing)
    info, ict = None, 0.0
    if dc is not None:
        info = digital.add_data_centres(n, temperature_on(year, n.snapshots), a, **dc)
    if "control" in other:
        ict = digital.add_digital_control(n, other["control"])
    if "home" in other:
        digital.add_work_at_home(n, *other["home"])
    n.generators_t.p_max_pu = n.generators_t.p_max_pu.where(n.generators_t.p_max_pu > 0.005, 0.0)
    n.optimize.create_model()
    base.tie_battery(n)
    if info and info["sited"]:
        digital.tie_siting(n, info)
    solution = solve.solve(n)
    zones = inputs.ZONES
    w = n.snapshot_weightings.generators
    energy = w
    # shadow price of the balance of each country, per MWh (the dual refers to a step that stands for several hours)
    price = solve.dual_of(n, solution, "Bus-nodal_balance")[zones].div(energy, axis=0)
    gen_p = solve.variable(n, solution, "Generator-p")
    link_p = solve.variable(n, solution, "Link-p")
    capacity = solve.variable(n, solution, "Generator-p_nom")
    links_nom = solve.variable(n, solution, "Link-p_nom")
    stores_nom = solve.variable(n, solution, "Store-e_nom")
    out = {"year": year, "scenario": name, "cost_bn": solution["objective"] / 1e9, "seconds": round(time.time() - t0),
           "backup_twh": float(gen_p[[f"{z} backup" for z in zones]].mul(w, axis=0).sum().sum() / 1e6),
           "shed_gwh": float(gen_p[[f"{z} shed" for z in zones]].mul(w, axis=0).sum().sum() / 1e3),
           "price_mean": float((price * n.loads_t.p_set[zones]).mul(w, axis=0).sum().sum()
                               / n.loads_t.p_set[zones].mul(w, axis=0).sum().sum()),
           "ict_twh": ict}
    for carrier in ("solar", "onwind", "offwind", "backup"):
        names = [x for x in capacity.index if x.endswith(f" {carrier}")]
        out[f"{carrier}_gw"] = float(capacity[names].sum() / 1e3)
    out["battery_gwh"] = float(stores_nom[[x for x in stores_nom.index if x.endswith("battery store") and "site" not in x]].sum() / 1e3)
    if name == "no new digital demand":
        # what one more MW of flat IT load would pay in each country: price times cooling overhead
        cooling = digital.pue(temperature_on(year, n.snapshots)[zones], a)
        out["marginal_cost_by_country"] = {z: float((price[z] * cooling[z]).mean()) for z in zones}
        out["price_by_country"] = {z: float(price[z].mean()) for z in zones}
        out["mean_pue"] = {z: float(cooling[z].mean()) for z in zones}
    if info:
        flows = link_p[info["links"]].mul(w, axis=0).sum() / 1e6
        interactive = n.loads_t.p_set[[f"{z} dc interactive" for z in info["zones"]]].mul(w, axis=0).sum().sum() / 1e6
        out["dc_twh"] = float(flows.sum() + interactive)
        out["batch_twh_by_country"] = {k.split()[0]: float(v) for k, v in flows.items()}
        out["compute_gw_by_country"] = {z: float((links_nom[f"{z} compute"] if info["sited"]
                                                  else n.links.at[f"{z} compute", "p_nom"]) / 1e3)
                                        for z in info["zones"]}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(year, name, "cost", round(out["cost_bn"], 3), "s", out["seconds"], flush=True)
    return out


def report() -> None:
    runs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((RES / "runs").glob("*.json"))]
    d = pd.DataFrame(runs)
    d["scenario"] = pd.Categorical(d.scenario, list(SCENARIOS), ordered=True)
    d = d.sort_values(["year", "scenario"])
    base_cost = d[d.scenario == "no new digital demand"].set_index("year").cost_bn
    base_backup = d[d.scenario == "no new digital demand"].set_index("year").backup_twh
    d["added_cost_bn"] = d.cost_bn - d.year.map(base_cost)
    d["added_backup_twh"] = d.backup_twh - d.year.map(base_backup)
    d["eur_per_mwh_dc"] = 1000 * d.added_cost_bn / d.dc_twh
    cols = ["year", "scenario", "cost_bn", "added_cost_bn", "eur_per_mwh_dc", "dc_twh", "added_backup_twh",
            "price_mean", "solar_gw", "onwind_gw", "offwind_gw", "backup_gw", "battery_gwh", "shed_gwh", "ict_twh"]
    d[cols].to_csv(RES / "scenarios.csv", index=False, float_format="%.5g")
    # means only over the scenarios solved for every weather year; the rest are compared within the first year
    key = d[d.scenario.isin(KEY)]
    mean = key[cols].groupby("scenario", observed=True).mean(numeric_only=True).drop(columns="year")
    spread = key.groupby("scenario", observed=True).added_cost_bn.agg(["min", "max"])
    mean = mean.join(spread.rename(columns={"min": "added_cost_min", "max": "added_cost_max"}))
    mean["weather_years"] = key.groupby("scenario", observed=True).year.nunique()
    mean.to_csv(RES / "scenarios_mean.csv", float_format="%.5g")
    print(mean.T.to_string())
    spatial = []
    for r in runs:
        if r["scenario"] == "no new digital demand":
            spatial += [{"year": r["year"], "country": z, "marginal_eur_per_mwh_it": v,
                         "price": r["price_by_country"][z], "pue": r["mean_pue"][z]}
                        for z, v in r["marginal_cost_by_country"].items()]
    pd.DataFrame(spatial).to_csv(RES / "marginal_cost_by_country.csv", index=False, float_format="%.5g")
    sited = [{"year": r["year"], "scenario": r["scenario"], "country": z, "compute_gw": v,
              "batch_twh": r["batch_twh_by_country"][z]}
             for r in runs if "compute_gw_by_country" in r for z, v in r["compute_gw_by_country"].items()]
    pd.DataFrame(sited).to_csv(RES / "siting.csv", index=False, float_format="%.5g")
    digital.times_tables().to_csv(RES / "times_process_tables.csv", index=False, float_format="%.5g")
    FIG.mkdir(exist_ok=True)
    first = d[(d.year == YEARS[0]) & d.eur_per_mwh_dc.notna()]
    names = [str(s) for s in first.scenario]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(names, first.eur_per_mwh_dc, color="#1f77b4", label=f"weather year {YEARS[0]}")
    for year, marker in zip(YEARS[1:], ("o", "D")):
        other = d[(d.year == year) & d.eur_per_mwh_dc.notna()]
        ax.plot(other.eur_per_mwh_dc, [str(s) for s in other.scenario], marker, color="black", ms=5, ls="none",
                label=f"weather year {year}")
    ax.invert_yaxis()
    ax.set_xlabel("added system cost per MWh of data-centre electricity, EUR")
    ax.set_xlim(0, 100)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.4, 1.12), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "cost_per_mwh.png", dpi=150)
    plt.close(fig)
    placed = pd.DataFrame(sited)
    placed = placed[(placed.year == YEARS[0])]
    placed = placed.pivot(index="country", columns="scenario", values="compute_gw")
    placed = placed[["batch may wait 24 h", "24 h, sited by the model", "24 h, sited, waste heat reused"]]
    placed.columns = ["assumed national shares", "sited by the model", "sited, waste heat reused"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    placed.sort_values("assumed national shares", ascending=False).plot.bar(ax=ax)
    ax.set(ylabel="compute capacity for batch work, GW", xlabel="", title=f"Siting, weather year {YEARS[0]}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "siting.png", dpi=150)
    plt.close(fig)
    m = pd.DataFrame(spatial).pivot(index="country", columns="year", values="marginal_eur_per_mwh_it")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    m.loc[m.mean(axis=1).sort_values().index].plot.bar(ax=ax)
    ax.set(ylabel="EUR per MWh of IT load", xlabel="", title="Marginal system cost of flat IT load before any is added")
    ax.legend(frameon=False, title="weather year")
    fig.tight_layout()
    fig.savefig(FIG / "marginal_cost_by_country.png", dpi=150)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["report"]:
        report()
    else:
        data = setting()
        for year in [int(a) for a in args] or YEARS:
            for scenario in (SCENARIOS if year == YEARS[0] else KEY):
                run_one(year, scenario, data)
            report()
