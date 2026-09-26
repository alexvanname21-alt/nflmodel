"""
Thin client for The Odds API, scoped to NFL only.

Docs: https://the-odds-api.com/liveapi/guides/v4/

Two-tier usage, because player props are only available on the per-event
endpoint (much more expensive in credits than the bulk odds endpoint):
  - get_game_odds()   -> h2h/spreads/totals for every upcoming game, 1 call.
  - get_player_props(event_id) -> props for ONE game, 1 call per market group.
Only pull props for games you're actually going to show, not the full slate,
to avoid burning quota.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import requests

from config import NFL_SPORT_KEY, ODDS_API_BASE, ODDS_API_KEY

# Player prop market keys The Odds API exposes for NFL (per-event only).
PLAYER_PROP_MARKETS = [
    "player_pass_yds",
    "player_pass_tds",
    "player_pass_interceptions",
    "player_pass_completions",
    "player_pass_attempts",
    "player_rush_yds",
    "player_rush_attempts",
    "player_reception_yds",
    "player_receptions",
    "player_anytime_td",
]

GAME_MARKETS = ["h2h", "spreads", "totals"]

# FanDuel and DraftKings only, since that's what the user actually bets on.
BOOKMAKERS = ["fanduel", "draftkings"]


class OddsAPIError(Exception):
    pass


class OddsClient:
    def __init__(self, api_key: str | None = None, max_retries: int = 3, timeout: int = 15):
        self.api_key = api_key or ODDS_API_KEY
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not set. Add it to .env.")
        self.max_retries = max_retries
        self.timeout = timeout
        self.last_quota_remaining: int | None = None
        self.last_quota_used: int | None = None

    def _get(self, path: str, params: dict) -> list | dict:
        params = dict(params)
        params["apiKey"] = self.api_key
        url = f"{ODDS_API_BASE}{path}"

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = requests.get(url, params=params, timeout=self.timeout)
                remaining = resp.headers.get("x-requests-remaining")
                used = resp.headers.get("x-requests-used")
                if remaining is not None:
                    self.last_quota_remaining = int(remaining)
                if used is not None:
                    self.last_quota_used = int(used)

                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code == 401:
                    raise OddsAPIError("Invalid API key (401).")
                if resp.status_code == 422:
                    raise OddsAPIError(f"Bad request params (422): {resp.text}")
                if resp.status_code == 429:
                    wait = 30 * (attempt + 1)
                    time.sleep(wait)
                    continue
                raise OddsAPIError(f"Unexpected status {resp.status_code}: {resp.text}")
            except requests.exceptions.RequestException as e:
                last_err = str(e)
                time.sleep(2 ** attempt)
        raise OddsAPIError(f"All retries failed: {last_err}")

    def get_upcoming_games(self, sport_key: str = NFL_SPORT_KEY) -> list[dict]:
        """List of upcoming NFL events with id/teams/commence_time (0 credits)."""
        return self._get(f"/v4/sports/{sport_key}/events", {"dateFormat": "iso"})

    def get_game_odds(self, sport_key: str = NFL_SPORT_KEY) -> list[dict]:
        """h2h/spreads/totals for every upcoming game from FanDuel + DraftKings."""
        return self._get(
            f"/v4/sports/{sport_key}/odds",
            {
                "regions": "us",
                "markets": ",".join(GAME_MARKETS),
                "bookmakers": ",".join(BOOKMAKERS),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )

    def get_player_props(self, event_id: str, markets: list[str] | None = None) -> dict:
        """Player prop odds for a single event. Costs credits per market group."""
        markets = markets or PLAYER_PROP_MARKETS
        return self._get(
            f"/v4/sports/{NFL_SPORT_KEY}/events/{event_id}/odds",
            {
                "regions": "us",
                "markets": ",".join(markets),
                "bookmakers": ",".join(BOOKMAKERS),
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
        )

    def get_scores(self, days_from: int = 3, sport_key: str = NFL_SPORT_KEY) -> list[dict]:
        """Completed/live scores, used to grade past picks."""
        return self._get(
            f"/v4/sports/{sport_key}/scores",
            {"daysFrom": days_from, "dateFormat": "iso"},
        )
