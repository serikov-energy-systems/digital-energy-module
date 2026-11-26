"""The TIMES data file (no GAMS needed)."""

import numpy as np
import pandas as pd
import pytest

from digienergy import digital, times_link

FRACTIONS = {"SD": 0.25, "SN": 0.23, "WD": 0.25, "WN": 0.27}


def warm_days():
    index = pd.date_range("2011-01-01", "2011-12-31 23:00", freq="h")
    summer = np.isin(index.month, times_link.SUMMER)
    day = np.isin(index.hour, times_link.DAY)
    return pd.Series(8.0 + 10.0 * summer + 6.0 * day, index=index)


def test_slices_split_the_year_and_nights_are_cheaper_to_cool():
    t = times_link.slice_table(warm_days(), FRACTIONS, digital.Assumptions())
    assert t.fr_interactive.sum() == pytest.approx(1.0)
    assert t.fr_batch.sum() == pytest.approx(1.0)
    assert t.pue["SN"] < t.pue["SD"]
    assert t.pue["WN"] == pytest.approx(1.15)


def test_data_file_links_every_input_to_its_output():
    a = digital.Assumptions()
    table = times_link.slice_table(warm_days(), FRACTIONS, a)
    tables = {"REG1": table, "REG2": table}
    flex = times_link.write_dd(tables, {"REG1": 100.0, "REG2": 50.0}, a, flexible=True)
    rigid = times_link.write_dd(tables, {"REG1": 100.0, "REG2": 50.0}, a, flexible=False)
    for process in ("DCINT", "DCBAT", "DCUSE"):
        assert f"REG1.2030.{process}.ACT.SD" in flex
    assert "PRC_STGTSS" in flex and "DCSTG" not in rigid
    assert "REG2.2030.ITSBAT 20" in flex                       # 40 % of 50 PJ
    assert f"REG1.2030.DCBAT.SD.UP {1.5 * 0.4 * 100 * table.fr_batch['SD']:.6g}" in flex
