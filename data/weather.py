"""
Game-time weather forecast via the National Weather Service API (free, no key,
US-only — which is fine, every NFL stadium is in the US).

NWS only forecasts ~7 days out, so this is only useful once a game is within
that window. Earlier than that, get_game_weather() returns None and callers
should treat weather as "unknown / not yet in play" rather than "clear."
"""
from __future__ import annotations

from datetime import datetime, timezone

import requests

from data.stadiums import get_stadium, is_weather_relevant

NWS_BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "nfl-model (personal betting research tool)"}


def _closest_period(periods: list[dict], target: datetime) -> dict | None:
    best, best_diff = None, None
    for p in periods:
        start = datetime.fromisoformat(p["startTime"])
        diff = abs((start - target).total_seconds())
        if best_diff is None or diff < best_diff:
            best, best_diff = p, diff
    return best


def get_game_weather(home_team: str, kickoff_utc: datetime) -> dict | None:
    """
    Returns a dict with temp_f, wind_mph, wind_dir, precip_pct, short_forecast,
    dome (bool). Returns {"dome": True} immediately for indoor stadiums without
    hitting the network. Returns None if the game is outside the ~7 day NWS
    forecast window or the lookup fails.
    """
    stadium = get_stadium(home_team)
    if stadium is None:
        return None
    if not is_weather_relevant(home_team):
        return {"dome": True}

    if kickoff_utc.tzinfo is None:
        kickoff_utc = kickoff_utc.replace(tzinfo=timezone.utc)
    days_out = (kickoff_utc - datetime.now(timezone.utc)).days
    if days_out > 7:
        return None

    try:
        points_resp = requests.get(
            f"{NWS_BASE}/points/{stadium['lat']},{stadium['lon']}",
            headers=HEADERS,
            timeout=10,
        )
        points_resp.raise_for_status()
        forecast_url = points_resp.json()["properties"]["forecastHourly"]

        forecast_resp = requests.get(forecast_url, headers=HEADERS, timeout=10)
        forecast_resp.raise_for_status()
        periods = forecast_resp.json()["properties"]["periods"]
    except (requests.exceptions.RequestException, KeyError, ValueError):
        return None

    period = _closest_period(periods, kickoff_utc)
    if period is None:
        return None

    wind_speed = period.get("windSpeed", "0 mph")
    try:
        wind_mph = int("".join(c for c in wind_speed.split()[0] if c.isdigit()))
    except (ValueError, IndexError):
        wind_mph = 0

    return {
        "dome": False,
        "temp_f": period.get("temperature"),
        "wind_mph": wind_mph,
        "wind_dir": period.get("windDirection"),
        "precip_pct": period.get("probabilityOfPrecipitation", {}).get("value") or 0,
        "short_forecast": period.get("shortForecast"),
        "forecast_time": period.get("startTime"),
    }


def weather_total_adjustment(weather: dict | None) -> float:
    """
    Points to subtract from a projected total based on weather. Deliberately
    small — most of this is already priced into the market, this just nudges
    the model's independent number for genuinely extreme conditions.
    """
    if weather is None or weather.get("dome"):
        return 0.0

    adj = 0.0
    wind = weather.get("wind_mph") or 0
    if wind >= 20:
        adj -= 3.0
    elif wind >= 15:
        adj -= 1.5

    precip = weather.get("precip_pct") or 0
    if precip >= 60:
        adj -= 1.0

    temp = weather.get("temp_f")
    if temp is not None and temp <= 20:
        adj -= 1.0

    return adj
