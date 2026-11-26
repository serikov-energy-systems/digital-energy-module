"""Tests on a two-country toy system (no data download)."""

import numpy as np
import pandas as pd
import pypsa
import pytest

from digienergy import digital, solve


def toy(snapshots=48, step=3):
    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-07", periods=snapshots, freq=f"{step}h"))
    n.snapshot_weightings.loc[:, :] = float(step)
    sun = np.clip(np.sin((n.snapshots.hour - 6) / 12 * np.pi), 0, None)
    for z, scale in (("DE", 1.0), ("SE", 0.4)):
        n.add("Bus", z, carrier="AC")
        n.add("Load", z, bus=z, p_set=pd.Series(30000.0 * scale, index=n.snapshots))
        n.add("Generator", f"{z} solar", bus=z, p_max_pu=pd.Series(sun, index=n.snapshots), capital_cost=40000.0,
              p_nom_extendable=True)
        n.add("Generator", f"{z} backup", bus=z, p_nom_extendable=True, capital_cost=50000.0,
              marginal_cost=200.0 if z == "DE" else 120.0)
    n.add("Link", "DE-SE", bus0="DE", bus1="SE", p_nom=3000.0, p_min_pu=-1.0)
    return n


def temperature(n):
    t = pd.DataFrame(index=n.snapshots)
    t["DE"] = 22.0
    t["SE"] = 5.0
    return t


A = digital.Assumptions(demand_twh=40.0, shares={"DE": 0.7, "SE": 0.3}, district_heat={"DE": 0.1, "SE": 0.6},
                        max_site_share=0.8)


def run(n, info=None):
    n.optimize.create_model()
    if info and info["sited"]:
        digital.tie_siting(n, info)
    return solve.solve(n)


def test_pue_rises_above_the_free_cooling_limit():
    n = toy()
    p = digital.pue(temperature(n), A)
    assert p["SE"].iloc[0] == pytest.approx(1.15)
    assert p["DE"].iloc[0] == pytest.approx(1.15 + 0.012 * 7)


def test_clarabel_agrees_with_highs_on_cost_and_prices():
    n = toy()
    own = run(n)
    other = toy()
    other.optimize(solver_name="highs", output_flag=False)
    assert own["objective"] == pytest.approx(other.objective, rel=1e-5)
    price = solve.dual_of(n, own, "Bus-nodal_balance")[["DE", "SE"]]
    reference = other.buses_t.marginal_price[["DE", "SE"]]
    weighted = lambda p, net: float((p.div(net.snapshot_weightings.objective, axis=0)).mean().mean())  # noqa: E731
    assert weighted(price, n) == pytest.approx(weighted(reference * other.snapshot_weightings.objective.iloc[0], other),
                                               rel=2e-2)


def test_data_centre_energy_and_flexibility_lower_cost():
    costs = {}
    for name, kw in (("none", None), ("fixed", {}), ("wait", {"window_h": 12}), ("pooled", {"window_h": 12, "pooled": True}),
                     ("sited", {"window_h": 12, "sited": True})):
        n = toy()
        info = digital.add_data_centres(n, temperature(n), A, **kw) if kw is not None else None
        s = run(n, info)
        costs[name] = s["objective"]
        if name == "fixed":
            flows = solve.variable(n, s, "Link-p")[info["links"]]
            work = (flows / digital.pue(temperature(n), A)[info["zones"]].to_numpy()).mean()
            assert work.to_numpy() == pytest.approx((A.batch_share * info["it_mw"]).to_numpy(), rel=1e-4)
        if name == "sited":
            built = solve.variable(n, s, "Link-p_nom")[info["links"]]
            assert built.sum() == pytest.approx(info["total_capacity"], rel=1e-5)
            assert built.max() <= A.max_site_share * info["total_capacity"] * (1 + 1e-5)
    assert costs["fixed"] > costs["none"]
    assert costs["wait"] <= costs["fixed"] * (1 + 1e-6)
    assert costs["pooled"] <= costs["wait"] * (1 + 1e-6)


def test_on_site_batteries_cannot_raise_the_cost():
    costs = []
    for hours in (0.0, 2.0):
        n = toy()
        n.generators.loc[["DE solar", "SE solar"], "capital_cost"] = 1000.0    # surplus at noon, something to store
        digital.add_data_centres(n, temperature(n), A, window_h=12, battery_h=hours)
        costs.append(run(n)["objective"])
    assert costs[1] < costs[0]


def test_digital_control_and_work_at_home_change_demand():
    n = toy()
    ict = digital.add_digital_control(n, 0.02, ict_twh_per_percent=0.25)
    assert ict == pytest.approx(0.5)
    base = run(toy())["objective"]
    assert run(n)["objective"] != pytest.approx(base, rel=1e-9)
    n = toy()
    digital.add_work_at_home(n)
    shape = n.loads_t.p_set["DE home work"]
    assert shape.min() < 0 < shape.max()


def test_times_tables_cover_all_countries():
    t = digital.times_tables()
    assert len(t) == 2 * len(digital.SHARES)
    assert t.demand_pj.sum() == pytest.approx(130.0 / 1.25 * 3.6, rel=1e-6)
