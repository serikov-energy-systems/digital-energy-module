"""A greenfield-renewables capacity expansion of northwestern Europe for one weather year, in PyPSA
(the base model of github.com/serikov-energy-systems/eu-weather-years-pypsa).

Nodes are countries, with a transport model between them. Nuclear and hydro are existing. Wind, solar and batteries
(hydrogen storage only with DIGI_HYDROGEN=1) are chosen by the optimiser, and a costly dispatchable backup and load
shedding close the balance. A year is represented by every fourth week (DIGI_WEEKS), solved by digienergy.solve.
With `fleet` given, capacities are fixed and only dispatch is solved.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pypsa

from . import inputs

STEP_HOURS = int(os.environ.get("WEATHER_STEP_HOURS", 3))
VOLL = 8000.0            # EUR/MWh, assumed
BACKUP_FUEL = float(os.environ.get("WEATHER_BACKUP_FUEL", 200.0))  # EUR/MWh electric, assumed price of low-carbon fuel
RESERVOIR_HOURS = 1500.0 # assumed storage energy per MW of reservoir hydro
PHS_HOURS = 6.0
# hydrogen storage is never built at these costs in the base study; left out here to keep the problem small
HYDROGEN = bool(int(os.environ.get("DIGI_HYDROGEN", 0)))


def weather_year(cf: pd.DataFrame, year: int, step: int = STEP_HOURS) -> pd.DataFrame:
    """Capacity factors of one weather year on a non-leap calendar, averaged to the model step."""
    d = inputs.no_leap(cf[cf.index.year == year])
    return inputs.no_leap(d.resample(f"{step}h").mean())


WEEKS = int(os.environ.get("DIGI_WEEKS", 13))    # representative weeks per year (52: all but the last day)


def representative(frame: pd.DataFrame, step: int = STEP_HOURS, weeks: int = WEEKS) -> pd.DataFrame:
    """Every (52 / weeks)-th week of a frame given at the model step, in calendar order."""
    per_week = 168 // step
    stride = 52 // weeks
    week = np.arange(len(frame)) // per_week
    keep = (week < 52) & (week % stride == stride // 2) & (week // stride < weeks)
    return frame[keep]


def demand_on(index: pd.DatetimeIndex, demand: pd.DataFrame, step: int = STEP_HOURS) -> pd.DataFrame:
    """The demand year averaged to the step, placed on the calendar of the weather year by position."""
    d = demand.resample(f"{step}h").mean().dropna(how="all")
    d = d.iloc[: len(index)]
    if len(d) != len(index):
        raise ValueError(f"demand has {len(d)} steps, weather {len(index)}")
    return d.set_axis(index)


def profile_columns(cf: pd.DataFrame, zone: str) -> dict[str, str]:
    """Column per carrier for a zone. Countries without a split use the national wind series as onshore."""
    out = {"solar": f"{zone}_pv_national_current"}
    on, nat, off = f"{zone}_wind_onshore_current", f"{zone}_wind_national_current", f"{zone}_wind_offshore_current"
    out["onwind"] = on if on in cf else nat
    if off in cf:
        out["offwind"] = off
    return out


def build(cf: pd.DataFrame, demand: pd.DataFrame, costs: pd.DataFrame, limits: pd.DataFrame, fleet_existing: pd.DataFrame,
          zones: list[str] | None = None, fleet: dict | None = None, step: int = STEP_HOURS) -> pypsa.Network:
    zones = zones or list(inputs.ZONES)
    n = pypsa.Network()
    n.set_snapshots(cf.index)
    # a reduced year: each step stands for 8760 / (steps x step length) times its duration in cost and energy,
    # storage fills and empties over the chosen weeks in sequence
    scale = 8760.0 / (len(cf) * step)
    n.snapshot_weightings.loc[:, :] = float(step)
    n.snapshot_weightings.loc[:, ["objective", "generators"]] = float(step * scale)
    if len(cf) * step < 8736:
        full = pd.date_range(cf.index[0].normalize().replace(month=1, day=1), periods=8760 // step, freq=f"{step}h")
        full = full[: len(demand.resample(f"{step}h").mean().dropna(how="all"))]
        dem = representative(demand_on(full, demand, step), step).iloc[: len(cf)].set_axis(cf.index)
    else:
        dem = demand_on(cf.index, demand, step)
    fixed = fleet or {}

    def cap(name: str) -> dict:
        """Extendable with bounds, or fixed at the given fleet."""
        if fleet is not None:
            return {"p_nom": fixed.get(name, 0.0), "p_nom_extendable": False}
        return {"p_nom_extendable": True}

    for z in zones:
        n.add("Bus", z, carrier="AC")
        n.add("Load", z, bus=z, p_set=dem[z])
        n.add("Generator", f"{z} shed", bus=z, carrier="shed", marginal_cost=VOLL, p_nom=float(dem[z].max()))
        n.add("Generator", f"{z} backup", bus=z, carrier="backup", marginal_cost=BACKUP_FUEL,
              capital_cost=costs.loc["OCGT", "capital_cost"], **cap(f"{z} backup"))
        cols = profile_columns(cf, z)
        limit = {"solar": 1500.0 * inputs.LAND_KM2[z], "onwind": 1000.0 * inputs.LAND_KM2[z], "offwind": 1000.0 * inputs.OFFSHORE_GW.get(z, 0)}
        tech = {"solar": "solar-utility", "onwind": "onwind", "offwind": "offwind"}
        for carrier, col in cols.items():
            if limit[carrier] <= 0:
                continue
            kw = cap(f"{z} {carrier}")
            if fleet is None:
                kw["p_nom_max"] = limit[carrier]
            n.add("Generator", f"{z} {carrier}", bus=z, carrier=carrier, p_max_pu=cf[col].clip(0, 1).fillna(0.0),
                  capital_cost=costs.loc[tech[carrier], "capital_cost"], **kw)
        existing = fleet_existing.loc[z]
        if existing.get("Nuclear", 0) > 0:
            n.add("Generator", f"{z} nuclear", bus=z, carrier="nuclear", p_nom=existing["Nuclear"],
                  p_max_pu=inputs.NUCLEAR_AVAILABILITY, marginal_cost=10.0)
        if existing.get("Run-Of-River", 0) > 0:
            n.add("Generator", f"{z} ror", bus=z, carrier="ror", p_nom=existing["Run-Of-River"], p_max_pu=inputs.HYDRO_CF["Run-Of-River"])
        if existing.get("Reservoir", 0) > 0:
            p = existing["Reservoir"]
            n.add("StorageUnit", f"{z} reservoir", bus=z, carrier="reservoir", p_nom=p, max_hours=RESERVOIR_HOURS, p_min_pu=0.0,
                  inflow=p * inputs.HYDRO_CF["Reservoir"], efficiency_dispatch=1.0, efficiency_store=1.0, cyclic_state_of_charge=True)
        if existing.get("Pumped Storage", 0) > 0:
            n.add("StorageUnit", f"{z} phs", bus=z, carrier="phs", p_nom=existing["Pumped Storage"], max_hours=PHS_HOURS,
                  efficiency_store=0.87, efficiency_dispatch=0.87, cyclic_state_of_charge=True)
        add_storage(n, z, costs, fixed, fleet is not None, hydrogen=HYDROGEN)
    for _, r in limits.iterrows():
        if r["a"] in zones and r["b"] in zones:
            n.add("Link", f"{r['a']}-{r['b']} {r['kind']}", bus0=r["a"], bus1=r["b"], p_nom=float(r["mw"]), p_min_pu=-1.0, efficiency=1.0, carrier=r["kind"])
    return n


def add_storage(n: pypsa.Network, z: str, costs: pd.DataFrame, fixed: dict, is_fixed: bool,
                hydrogen: bool = True) -> None:
    """Battery and hydrogen storage: energy stores behind charge and discharge links."""
    def link_cap(name):
        return {"p_nom": fixed.get(name, 0.0), "p_nom_extendable": False} if is_fixed else {"p_nom_extendable": True}

    def store_cap(name):
        return {"e_nom": fixed.get(name, 0.0), "e_nom_extendable": False} if is_fixed else {"e_nom_extendable": True}

    n.add("Bus", f"{z} battery", carrier="battery")
    n.add("Store", f"{z} battery store", bus=f"{z} battery", e_cyclic=True, capital_cost=costs.loc["battery storage", "capital_cost"], **store_cap(f"{z} battery store"))
    eta = float(costs.loc["battery inverter", "efficiency"])
    n.add("Link", f"{z} battery charge", bus0=z, bus1=f"{z} battery", efficiency=eta, capital_cost=costs.loc["battery inverter", "capital_cost"], **link_cap(f"{z} battery charge"))
    n.add("Link", f"{z} battery discharge", bus0=f"{z} battery", bus1=z, efficiency=eta, **link_cap(f"{z} battery discharge"))
    if not hydrogen:
        return
    n.add("Bus", f"{z} H2", carrier="H2")
    n.add("Store", f"{z} H2 store", bus=f"{z} H2", e_cyclic=True, capital_cost=costs.loc["hydrogen storage underground", "capital_cost"], **store_cap(f"{z} H2 store"))
    n.add("Link", f"{z} electrolysis", bus0=z, bus1=f"{z} H2", efficiency=float(costs.loc["electrolysis", "efficiency"]),
          capital_cost=costs.loc["electrolysis", "capital_cost"], **link_cap(f"{z} electrolysis"))
    n.add("Link", f"{z} fuel cell", bus0=f"{z} H2", bus1=z, efficiency=float(costs.loc["fuel cell", "efficiency"]),
          capital_cost=costs.loc["fuel cell", "capital_cost"], **link_cap(f"{z} fuel cell"))


def tie_battery(n: pypsa.Network) -> None:
    """Discharge capacity equals charge capacity for batteries, in efficiency-free MW."""
    m = n.model
    for z in n.buses.index[n.buses.carrier == "AC"]:
        a, b = f"{z} battery charge", f"{z} battery discharge"
        if a in n.links.index and n.links.at[a, "p_nom_extendable"]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m.add_constraints(m["Link-p_nom"].loc[a] - m["Link-p_nom"].loc[b] == 0, name=f"tie-{z}")
