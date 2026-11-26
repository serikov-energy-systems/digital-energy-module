"""Write the data-centre processes of the module as a data file for the TIMES model generator.

The file targets the two-region open TIMES demo (github.com/etsap-TIMES/TIMES_Demo, time slices SD, SN, WD, WN):

  DCINT   electricity -> interactive IT service, demand with a fixed share per time slice
  DCBAT   electricity -> batch work, activity bounded per time slice by the installed compute capacity
  DCSTG   optional time slice storage of batch work (work done by night serves the day of the same season)
  DCUSE   batch work -> batch IT service

The efficiency of DCINT and DCBAT per time slice is 1 / PUE, with PUE from hourly temperature as in the PyPSA part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import digital

SLICES = ("SD", "SN", "WD", "WN")
SUMMER = (4, 5, 6, 7, 8, 9)
DAY = range(7, 19)
YEARS = (2005, 2030)


def slice_of(index: pd.DatetimeIndex) -> pd.Series:
    """Season (April to September is summer) and day (07:00 to 19:00) or night of each hour."""
    season = np.where(index.month.isin(SUMMER), "S", "W")
    part = np.where(index.hour.isin(DAY), "D", "N")
    return pd.Series(np.char.add(season.astype(str), part.astype(str)), index=index)


def slice_table(temperature: pd.Series, fractions: dict, a: digital.Assumptions) -> pd.DataFrame:
    """Per time slice: mean PUE, share of the interactive demand and share of the batch demand.

    fractions: share of the year of each time slice in the target model."""
    cooling = digital.pue(temperature.to_frame("z"), a)["z"]
    key = slice_of(temperature.index)
    swing = pd.Series(1.0 + a.interactive_swing * np.sin(2 * np.pi * (temperature.index.hour - 9) / 24),
                      index=temperature.index)
    yr = pd.Series(fractions, dtype=float).loc[list(SLICES)]
    t = pd.DataFrame({"year_fraction": yr})
    # electricity per unit of IT work, weighted with the interactive profile
    t["pue"] = (cooling * swing).groupby(key).sum() / swing.groupby(key).sum()
    interactive = yr * swing.groupby(key).mean()
    t["fr_interactive"] = interactive / interactive.sum()
    t["fr_batch"] = yr / yr.sum()
    return t


def write_dd(tables: dict, demand_pj: dict, a: digital.Assumptions, flexible: bool, first_year_pj: float = 1.0) -> str:
    """tables: region -> slice_table; demand_pj: region -> IT electricity at PUE 1 in the last year."""
    regions = list(tables)
    processes = ["DCINT", "DCBAT", "DCUSE"] + (["DCSTG"] if flexible else [])
    sets: dict[str, list[str]] = {k: [] for k in ("COM_TMAP", "COM_TSL", "COM_UNIT", "PRC_MAP", "PRC_TSL",
                                                  "PRC_ACTUNT", "TOP", "PRC_STGTSS")}
    par: dict[str, list[str]] = {k: [] for k in ("COM_PROJ", "COM_FR", "ACT_EFF", "ACT_BND")}
    topology = {"DCINT": ("DMD", "ITSINT", [("ELC", "IN"), ("ITSINT", "OUT")]),
                "DCBAT": ("PRE", "ITB", [("ELC", "IN"), ("ITB", "OUT")]),
                "DCUSE": ("DMD", "ITSBAT", [("ITB", "IN"), ("ITSBAT", "OUT")]),
                "DCSTG": ("STG", "ITB", [("ITB", "IN"), ("ITB", "OUT")])}
    for r in regions:
        t = tables[r]
        for com, kind in (("ITSINT", "DEM"), ("ITSBAT", "DEM"), ("ITB", "NRG")):
            sets["COM_TMAP"].append(f"{r}.{kind}.{com}")
            sets["COM_TSL"].append(f"{r}.{com}.DAYNITE")
            sets["COM_UNIT"].append(f"{r}.{com}.PJ")
        for p in processes:
            group, primary, flows = topology[p]
            sets["PRC_MAP"].append(f"{r}.{group}.{p}")
            sets["PRC_TSL"].append(f"{r}.{p}.DAYNITE")
            sets["PRC_ACTUNT"].append(f"{r}.{p}.{primary}.PJ")
            sets["TOP"] += [f"{r}.{p}.{c}.{io}" for c, io in flows]
        if flexible:
            sets["PRC_STGTSS"].append(f"{r}.DCSTG.ITB")
        total = float(demand_pj[r])
        for com, share, column in (("ITSINT", 1 - a.batch_share, "fr_interactive"), ("ITSBAT", a.batch_share, "fr_batch")):
            par["COM_PROJ"] += [f"{r}.{YEARS[0]}.{com} {first_year_pj * share:.6g}",
                                f"{r}.{YEARS[-1]}.{com} {total * share:.6g}"]
            par["COM_FR"] += [f"{r}.{y}.{com}.{s} {t.at[s, column]:.8f}" for y in YEARS for s in SLICES]
        for p in ("DCINT", "DCBAT"):
            par["ACT_EFF"] += [f"{r}.{y}.{p}.ACT.{s} {1 / t.at[s, 'pue']:.6f}" for y in YEARS for s in SLICES]
        # without an efficiency TIMES leaves the input of a process unrelated to its output
        par["ACT_EFF"] += [f"{r}.{y}.DCUSE.ACT.{s} 1" for y in YEARS for s in SLICES]
        for y, volume in ((YEARS[0], first_year_pj), (YEARS[-1], total)):
            par["ACT_BND"] += [f"{r}.{y}.DCBAT.{s}.UP {a.headroom * a.batch_share * volume * t.at[s, 'fr_batch']:.6g}"
                               for s in SLICES]
    lines = ["* Data-centre processes written by digienergy.times_link. All values are assumptions.",
             "SET COM_GRP\n/\nITSINT\nITSBAT\nITB\n/", "SET COM\n/\nITSINT\nITSBAT\nITB\n/",
             "SET PRC\n/\n" + "\n".join(processes) + "\n/"]
    for name, rows in sets.items():
        if rows:
            lines.append(f"SET {name}\n/\n" + "\n".join(rows) + "\n/")
    for name, rows in par.items():
        lines.append(f"PARAMETER\n{name} ' '/\n" + "\n".join(rows) + "\n/")
    return "\n\n".join(lines) + "\n"
