"""Hourly outdoor temperature (ERA5 through Open-Meteo, CC BY 4.0) at one data-centre location per country."""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests

SITES = {"DE": (50.11, 8.68), "FR": (48.86, 2.35), "BE": (50.85, 4.35), "NL": (52.37, 4.90), "AT": (48.21, 16.37),
         "CH": (47.38, 8.54), "DK": (55.68, 12.57), "NO": (59.91, 10.75), "SE": (59.33, 18.07), "GB": (51.51, -0.13),
         "PL": (52.23, 21.01), "CZ": (50.08, 14.44)}
URL = "https://archive-api.open-meteo.com/v1/archive"


def load_year(year: int, folder: Path) -> pd.DataFrame:
    """Hourly temperature (degrees Celsius) of one calendar year for all countries; one file per year, cached."""
    path = folder / f"temperature_{year}.csv"
    if path.exists():
        return pd.read_csv(path, index_col=0, parse_dates=True)
    params = {"latitude": ",".join(str(v[0]) for v in SITES.values()),
              "longitude": ",".join(str(v[1]) for v in SITES.values()),
              "start_date": f"{year}-01-01", "end_date": f"{year}-12-31", "hourly": "temperature_2m",
              "models": "era5", "timezone": "UTC"}
    for attempt in range(8):
        try:
            r = requests.get(URL, params=params, timeout=120, headers={"User-Agent": "digital-energy-module (research)"})
            if r.status_code == 200:
                break
            wait = 70 if r.status_code == 429 else 5 * (attempt + 1)
        except requests.RequestException:
            wait = 10 * (attempt + 1)
        time.sleep(wait)
    else:
        raise RuntimeError(f"temperature download failed for {year}")
    docs = r.json()
    out = pd.DataFrame({z: d["hourly"]["temperature_2m"] for z, d in zip(SITES, docs)},
                       index=pd.to_datetime(docs[0]["hourly"]["time"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path)
    return out
