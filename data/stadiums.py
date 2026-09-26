"""
Static NFL stadium reference table.

roof:
  "dome"        - fully enclosed, always climate controlled -> weather never matters
  "retractable"  - can be open or closed; we treat as outdoor-risk unless a team is
                   known to close it in bad weather (conservative default: treat like
                   outdoor for forecasting purposes, since we can't know roof status
                   in advance for most games)
  "outdoor"     - weather always in play

team abbreviations follow nfl_data_py convention.
"""

STADIUMS = {
    "ARI": {"name": "State Farm Stadium", "roof": "retractable", "lat": 33.5276, "lon": -112.2626, "surface": "grass"},
    "ATL": {"name": "Mercedes-Benz Stadium", "roof": "dome", "lat": 33.7554, "lon": -84.4008, "surface": "turf"},
    "BAL": {"name": "M&T Bank Stadium", "roof": "outdoor", "lat": 39.2780, "lon": -76.6227, "surface": "grass"},
    "BUF": {"name": "Highmark Stadium", "roof": "outdoor", "lat": 42.7738, "lon": -78.7870, "surface": "turf"},
    "CAR": {"name": "Bank of America Stadium", "roof": "outdoor", "lat": 35.2258, "lon": -80.8528, "surface": "grass"},
    "CHI": {"name": "Soldier Field", "roof": "outdoor", "lat": 41.8623, "lon": -87.6167, "surface": "grass"},
    "CIN": {"name": "Paycor Stadium", "roof": "outdoor", "lat": 39.0954, "lon": -84.5160, "surface": "turf"},
    "CLE": {"name": "Huntington Bank Field", "roof": "outdoor", "lat": 41.5061, "lon": -81.6995, "surface": "grass"},
    "DAL": {"name": "AT&T Stadium", "roof": "retractable", "lat": 32.7473, "lon": -97.0945, "surface": "turf"},
    "DEN": {"name": "Empower Field at Mile High", "roof": "outdoor", "lat": 39.7439, "lon": -105.0201, "surface": "grass"},
    "DET": {"name": "Ford Field", "roof": "dome", "lat": 42.3400, "lon": -83.0456, "surface": "turf"},
    "GB": {"name": "Lambeau Field", "roof": "outdoor", "lat": 44.5013, "lon": -88.0622, "surface": "grass"},
    "HOU": {"name": "NRG Stadium", "roof": "retractable", "lat": 29.6847, "lon": -95.4107, "surface": "turf"},
    "IND": {"name": "Lucas Oil Stadium", "roof": "retractable", "lat": 39.7601, "lon": -86.1639, "surface": "turf"},
    "JAX": {"name": "EverBank Stadium", "roof": "outdoor", "lat": 30.3239, "lon": -81.6373, "surface": "grass"},
    "KC": {"name": "GEHA Field at Arrowhead Stadium", "roof": "outdoor", "lat": 39.0489, "lon": -94.4839, "surface": "grass"},
    "LV": {"name": "Allegiant Stadium", "roof": "dome", "lat": 36.0909, "lon": -115.1833, "surface": "turf"},
    "LAC": {"name": "SoFi Stadium", "roof": "dome", "lat": 33.9535, "lon": -118.3392, "surface": "turf"},
    "LA": {"name": "SoFi Stadium", "roof": "dome", "lat": 33.9535, "lon": -118.3392, "surface": "turf"},
    "MIA": {"name": "Hard Rock Stadium", "roof": "outdoor", "lat": 25.9580, "lon": -80.2389, "surface": "grass"},
    "MIN": {"name": "U.S. Bank Stadium", "roof": "dome", "lat": 44.9737, "lon": -93.2577, "surface": "turf"},
    "NE": {"name": "Gillette Stadium", "roof": "outdoor", "lat": 42.0909, "lon": -71.2643, "surface": "turf"},
    "NO": {"name": "Caesars Superdome", "roof": "dome", "lat": 29.9509, "lon": -90.0815, "surface": "turf"},
    "NYG": {"name": "MetLife Stadium", "roof": "outdoor", "lat": 40.8135, "lon": -74.0745, "surface": "turf"},
    "NYJ": {"name": "MetLife Stadium", "roof": "outdoor", "lat": 40.8135, "lon": -74.0745, "surface": "turf"},
    "PHI": {"name": "Lincoln Financial Field", "roof": "outdoor", "lat": 39.9008, "lon": -75.1675, "surface": "grass"},
    "PIT": {"name": "Acrisure Stadium", "roof": "outdoor", "lat": 40.4468, "lon": -80.0158, "surface": "grass"},
    "SF": {"name": "Levi's Stadium", "roof": "outdoor", "lat": 37.4032, "lon": -121.9698, "surface": "grass"},
    "SEA": {"name": "Lumen Field", "roof": "outdoor", "lat": 47.5952, "lon": -122.3316, "surface": "turf"},
    "TB": {"name": "Raymond James Stadium", "roof": "outdoor", "lat": 27.9759, "lon": -82.5033, "surface": "grass"},
    "TEN": {"name": "Nissan Stadium", "roof": "outdoor", "lat": 36.1665, "lon": -86.7713, "surface": "grass"},
    "WAS": {"name": "Northwest Stadium", "roof": "outdoor", "lat": 38.9077, "lon": -76.8645, "surface": "grass"},
}


def is_weather_relevant(team_abbr: str) -> bool:
    """False for domes (and dome-equivalent retractables like SoFi/Allegiant)."""
    info = STADIUMS.get(team_abbr)
    if not info:
        return True
    return info["roof"] != "dome"


def get_stadium(team_abbr: str) -> dict | None:
    return STADIUMS.get(team_abbr)
