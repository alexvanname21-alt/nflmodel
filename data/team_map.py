"""Maps The Odds API's full team names to nflverse's team abbreviations."""

ODDS_NAME_TO_ABBR = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}

ABBR_TO_ODDS_NAME = {v: k for k, v in ODDS_NAME_TO_ABBR.items()}


def to_abbr(odds_team_name: str) -> str | None:
    return ODDS_NAME_TO_ABBR.get(odds_team_name)


def nickname(abbr: str) -> str:
    """'ARI' -> 'Cardinals', for terse human-readable picks."""
    full = ABBR_TO_ODDS_NAME.get(abbr)
    return full.split()[-1] if full else abbr


def fmt_spread(value: float | None) -> str:
    """5.0 -> '+5', -10.5 -> '-10.5' — always signed, no unnecessary decimals."""
    if value is None:
        return ""
    text = f"{value:+.1f}".rstrip("0").rstrip(".")
    return text if text not in ("+", "-") else "+0"
