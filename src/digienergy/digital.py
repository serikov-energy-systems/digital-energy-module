"""A digitalisation module for a country-level PyPSA energy system model.

Three kinds of digital elements can be attached to a network built by ``digienergy.base``:

  data centres        interactive work that follows its own country and hour, deferrable batch work that may wait
                      (a backlog store), move between countries (one common pool of work) or be sited by the model;
                      cooling through a temperature-dependent PUE; optional revenue from reusing waste heat
  digital control     a share of national demand that artificial-intelligence or automation tools can shift within
                      some hours, paid for with the electricity of the digital system itself
  digital practices   an hourly change of national demand, as from more work at home

All parameters are assumptions collected in ``Assumptions``; none of them is a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:          # the TIMES link uses this module without PyPSA installed
    import pypsa

# Added data-centre electricity by 2030 (IT plus cooling at the reference PUE), shares by country. Assumed.
SHARES = {"DE": 0.24, "GB": 0.23, "FR": 0.16, "NL": 0.08, "SE": 0.07, "NO": 0.06, "DK": 0.05, "CH": 0.04,
          "BE": 0.03, "PL": 0.02, "AT": 0.01, "CZ": 0.01}
# Share of recovered heat that a district heating system could take, by country. Assumed.
DISTRICT_HEAT = {"DE": 0.15, "GB": 0.03, "FR": 0.07, "NL": 0.07, "SE": 0.60, "NO": 0.25, "DK": 0.70, "CH": 0.10,
                 "BE": 0.03, "PL": 0.55, "AT": 0.35, "CZ": 0.45}
HEATING_MONTHS = (10, 11, 12, 1, 2, 3, 4)


@dataclass(frozen=True)
class Assumptions:
    demand_twh: float = 130.0           # added data-centre electricity per year at the reference PUE
    reference_pue: float = 1.25
    batch_share: float = 0.4            # share of the IT work that may be moved in time or place
    headroom: float = 1.5               # compute capacity for batch work relative to its average
    max_site_share: float = 0.30        # largest share of the batch capacity in one country when the model sites it
    interactive_swing: float = 0.10     # daily swing of interactive work around its mean
    pue_floor: float = 1.15
    pue_slope: float = 0.012            # per degree above the free cooling limit
    free_cooling_c: float = 15.0
    heat_value: float = 30.0            # EUR per MWh of heat delivered
    heat_recovery: float = 0.7          # share of the electricity that can be captured as heat
    battery_efficiency: float = 0.95    # on-site battery, each direction
    shares: dict = field(default_factory=lambda: dict(SHARES))
    district_heat: dict = field(default_factory=lambda: dict(DISTRICT_HEAT))


def pue(temperature: pd.DataFrame, a: Assumptions) -> pd.DataFrame:
    """Power usage effectiveness by country and step from the outdoor temperature."""
    return a.pue_floor + a.pue_slope * (temperature - a.free_cooling_c).clip(lower=0.0)


def it_load(a: Assumptions, zones: list[str]) -> pd.Series:
    """Average IT power per country (MW) implied by the yearly electricity at the reference PUE."""
    total = a.demand_twh * 1e6 / 8760.0 / a.reference_pue
    shares = pd.Series({z: a.shares.get(z, 0.0) for z in zones})
    return total * shares / shares.sum()


def add_data_centres(n: pypsa.Network, temperature: pd.DataFrame, a: Assumptions = Assumptions(),
                     window_h: float = 0.0, pooled: bool = False, sited: bool = False,
                     heat_reuse: bool = False, battery_h: float = 0.0) -> dict:
    """Attach data centres to the AC buses.

    window_h: how long batch work may wait; pooled: batch work of all countries can run anywhere with free capacity;
    sited: the model decides where the batch capacity is built (implies pooled); heat_reuse: revenue for waste heat in
    the heating months; battery_h: on-site batteries open to the grid, in hours of the average data-centre load (their
    cost is not counted, so the result is what they are worth to the system). Returns the names of the compute links
    and the average IT load."""
    zones = [z for z in n.buses.index[n.buses.carrier == "AC"] if a.shares.get(z, 0.0) > 0]
    snapshots = n.snapshots
    cooling = pue(temperature.loc[snapshots, zones], a)
    mean_it = it_load(a, zones)
    hour = snapshots.hour
    swing = 1.0 + a.interactive_swing * np.sin(2 * np.pi * (hour - 9) / 24)
    pooled = pooled or sited
    if pooled:
        n.add("Bus", "work", carrier="work")
        n.add("Load", "work", bus="work", p_set=float(a.batch_share * mean_it.sum()))
    links = []
    for z in zones:
        interactive = (1 - a.batch_share) * mean_it[z] * swing
        n.add("Load", f"{z} dc interactive", bus=z, p_set=pd.Series(interactive, index=snapshots) * cooling[z])
        batch = a.batch_share * mean_it[z]
        bus = "work"
        if not pooled:
            bus = f"{z} work"
            n.add("Bus", bus, carrier="work")
            n.add("Load", bus, bus=bus, p_set=float(batch))
            if window_h > 0:
                n.add("Store", f"{z} backlog", bus=bus, e_nom=float(window_h * batch), e_min_pu=-1.0, e_max_pu=0.0,
                      e_cyclic=True)
        capacity = a.headroom * batch * a.reference_pue            # MW of electricity
        kwargs = {}
        if heat_reuse:
            winter = np.isin(snapshots.month, HEATING_MONTHS)
            credit = a.heat_value * a.heat_recovery * a.district_heat.get(z, 0.0)
            kwargs["marginal_cost"] = pd.Series(np.where(winter, -credit, 0.0), index=snapshots)
        if sited:
            total = a.headroom * a.batch_share * mean_it.sum() * a.reference_pue
            kwargs.update(p_nom_extendable=True, p_nom_max=float(a.max_site_share * total))
        else:
            kwargs.update(p_nom=float(capacity))
        n.add("Link", f"{z} compute", bus0=z, bus1=bus, efficiency=1.0 / cooling[z], carrier="compute", **kwargs)
        links.append(f"{z} compute")
        if battery_h > 0:
            power = float(mean_it[z] * a.reference_pue)
            n.add("Bus", f"{z} site battery", carrier="battery")
            n.add("Store", f"{z} site battery store", bus=f"{z} site battery", e_nom=power * battery_h, e_cyclic=True)
            n.add("Link", f"{z} site charge", bus0=z, bus1=f"{z} site battery", p_nom=power, efficiency=a.battery_efficiency)
            n.add("Link", f"{z} site discharge", bus0=f"{z} site battery", bus1=z, p_nom=power,
                  efficiency=a.battery_efficiency)
    if pooled and window_h > 0:
        n.add("Store", "work backlog", bus="work", e_nom=float(window_h * a.batch_share * mean_it.sum()),
              e_min_pu=-1.0, e_max_pu=0.0, e_cyclic=True)
    return {"links": links, "zones": zones, "it_mw": mean_it, "sited": sited,
            "total_capacity": float(a.headroom * a.batch_share * mean_it.sum() * a.reference_pue)}


def tie_siting(n: pypsa.Network, info: dict) -> None:
    """The model places a fixed total of compute capacity (call after the optimisation model has been created)."""
    p = n.model["Link-p_nom"].loc[info["links"]]
    n.model.add_constraints(p.sum() == info["total_capacity"], name="compute-total")


def add_digital_control(n: pypsa.Network, share: float, hours: float = 8.0, ict_twh_per_percent: float = 0.25) -> float:
    """A share of each country's average demand can be shifted within `hours`; the digital system itself uses
    electricity (assumed per percentage point of shiftable share, spread like demand). Returns that electricity, TWh."""
    zones = list(n.buses.index[n.buses.carrier == "AC"])
    load = n.loads_t.p_set[zones]
    total = load.mean().sum()
    ict_twh = ict_twh_per_percent * share * 100
    for z in zones:
        mean = float(load[z].mean())
        power = share * mean
        n.add("Bus", f"{z} shift", carrier="shift")
        n.add("Store", f"{z} shift store", bus=f"{z} shift", e_nom=power * hours, e_cyclic=True)
        n.add("Link", f"{z} shift down", bus0=z, bus1=f"{z} shift", p_nom=power)
        n.add("Link", f"{z} shift up", bus0=f"{z} shift", bus1=z, p_nom=power)
        n.add("Load", f"{z} ict", bus=z, p_set=ict_twh * 1e6 / 8760.0 * mean / total)
    return ict_twh


def add_work_at_home(n: pypsa.Network, office: float = -0.010, home: float = 0.006) -> None:
    """More work at home: demand falls by `office` of the average on working days from 08:00 to 18:00 (offices,
    commuting) and rises by `home` of the average on working days from 06:00 to 23:00. Both shares are assumed."""
    zones = list(n.buses.index[n.buses.carrier == "AC"])
    s = n.snapshots
    working = s.dayofweek < 5
    day = working & (s.hour >= 8) & (s.hour < 18)
    awake = working & (s.hour >= 6) & (s.hour < 23)
    shape = pd.Series(office * day + home * awake, index=s)
    for z in zones:
        n.add("Load", f"{z} home work", bus=z, p_set=shape * float(n.loads_t.p_set[z].mean()))


def times_tables(a: Assumptions = Assumptions()) -> pd.DataFrame:
    """The data-centre processes described in the terms of a TIMES-type model (process, commodities, efficiency,
    bounds), as a starting point for a link to such models."""
    zones = list(a.shares)
    mean = it_load(a, zones)
    rows = []
    for z in zones:
        rows.append({"region": z, "process": f"DCINT_{z}", "commodity_in": "ELCC", "commodity_out": "ITSINT",
                     "efficiency": f"1/PUE(t), reference {a.reference_pue}", "demand_pj": mean[z] * (1 - a.batch_share) * 8760 * 3.6e-6,
                     "timeslice_flexibility": "none", "capacity_bound_gw": ""})
        rows.append({"region": z, "process": f"DCBAT_{z}", "commodity_in": "ELCC", "commodity_out": "ITSBAT",
                     "efficiency": f"1/PUE(t), reference {a.reference_pue}", "demand_pj": mean[z] * a.batch_share * 8760 * 3.6e-6,
                     "timeslice_flexibility": "storage process ITSBAT (day or week)",
                     "capacity_bound_gw": a.headroom * a.batch_share * mean[z] * a.reference_pue / 1000})
    return pd.DataFrame(rows)
