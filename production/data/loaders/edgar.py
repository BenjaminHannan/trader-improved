"""SEC EDGAR XBRL fundamentals with point-in-time filing vintages.

The single most important fact about fundamentals is that they are knowable *at
filing*, never at the fiscal period end. A company's Q1 (ending March 31) EPS is not
public until the 10-Q is filed weeks later; using it on March 31 is a look-ahead
fantasy that has flattered many a backtest to death. So this loader stamps
``available_from`` from the SEC ``filed`` date (at 21:00 UTC, comfortably after the
US market close on the filing day), NOT from the period end (``obs_date``). That lag
is the entire point of routing fundamentals through EDGAR rather than a vendor's
already-aligned quarterly file.

Source: the SEC ``companyfacts`` XBRL API
(``https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json``), with ticker->CIK
resolved from ``https://www.sec.gov/files/company_tickers.json``. The SEC *requires* a
descriptive ``User-Agent`` header carrying a contact email and rate-limits to 10 req/s;
we read the UA from ``SEC_USER_AGENT`` (falling back to a documented default) and pace
requests to <= 8 req/s to stay well inside the limit.

Concept extraction (first present in each fallback chain wins):
  * eps      us-gaap:EarningsPerShareDiluted  -> fallback us-gaap:EarningsPerShareBasic
  * revenue  us-gaap:Revenues                 -> fallback us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax
  * shares   dei:CommonStockSharesOutstanding -> fallback dei:EntityCommonStockSharesOutstanding
             -> fallback us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding

Vintages. XBRL re-reports the same period across many later filings (prior-year
comparatives) and, occasionally, *restates* it. We keep the FIRST filing of each
distinct value per (period, field) — a run-length dedup on the ``filed``-ordered
values — so the original release survives with its own filing date and a genuine
restatement coexists as a LATER vintage with the restatement's filing date. Nothing is
dropped: the lake and the availability-date as-of join downstream give exact PIT
semantics (a revision is visible only on/after ITS own ``available_from``).
"""
from __future__ import annotations

import os
import time

import pandas as pd

from production.data.base import AvailabilityRule, BaseLoader

# Contact UA is mandatory per SEC fair-access policy; override via env in production.
DEFAULT_USER_AGENT = "trader-improved research benjamin.a.hannan@gmail.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
COMPANYFACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
MIN_INTERVAL = 1.0 / 8.0  # <= 8 req/s pacing, inside the SEC 10 req/s limit

# field -> ordered (taxonomy, concept) fallback chain; first present & non-empty wins.
CONCEPTS: dict[str, list[tuple[str, str]]] = {
    "eps": [
        ("us-gaap", "EarningsPerShareDiluted"),
        ("us-gaap", "EarningsPerShareBasic"),
    ],
    "revenue": [
        ("us-gaap", "Revenues"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax"),
    ],
    "shares": [
        ("dei", "CommonStockSharesOutstanding"),
        ("dei", "EntityCommonStockSharesOutstanding"),
        ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
    ],
}

# Only periodic reports carry the numbers we want; amendments (10-K/A, 10-Q/A) are the
# natural home of restatements and are kept — they carry later `filed` dates.
_ALLOWED_FORM_PREFIXES = ("10-K", "10-Q")


class EdgarFundamentalsLoader(BaseLoader):
    dataset = "fundamentals"
    vendor = "sec"
    source = "sec:companyfacts"
    asset_classes = ["equity"]
    default_asset_class = "equity"
    # available_from is the filing timestamp built in transform (filed date @ 21:00 UTC).
    availability_rule = AvailabilityRule("explicit", {"column": "filed_ts"})
    expectations = {
        "columns": ["value"],
        "max_null_frac": 0.05,
        "min_rows": 1,
    }

    def __init__(self, lake=None, instruments=None, symbols=None, user_agent=None):
        super().__init__(lake, instruments)
        self.symbols = list(symbols or [])
        self.user_agent = user_agent or os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
        self._last_req = 0.0

    # ------------------------------------------------------------------ fetch
    def _throttle(self) -> None:
        """Space requests to honour the SEC rate limit (<= 8 req/s)."""
        wait = MIN_INTERVAL - (time.monotonic() - self._last_req)
        if wait > 0:
            time.sleep(wait)
        self._last_req = time.monotonic()

    def _get_json(self, url: str):
        import requests

        self._throttle()
        resp = requests.get(url, headers={"User-Agent": self.user_agent,
                                          "Accept-Encoding": "gzip, deflate"}, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _ticker_to_cik(self) -> dict[str, int]:
        """Map upper-cased ticker -> integer CIK from the SEC ticker directory."""
        data = self._get_json(TICKERS_URL)
        rows = data.values() if isinstance(data, dict) else data
        out: dict[str, int] = {}
        for rec in rows:
            tick = str(rec.get("ticker", "")).upper()
            cik = rec.get("cik_str", rec.get("cik"))
            if tick and cik is not None:
                out[tick] = int(cik)
        return out

    def fetch(self, start, end) -> dict[str, dict]:
        """Return ``{symbol: companyfacts_json}`` for every resolvable symbol.

        ``start``/``end`` are ignored at fetch time — companyfacts returns the full
        filing history for a CIK; the PIT filtering happens downstream via
        ``available_from``. Symbols that don't resolve to a CIK are skipped politely.
        """
        cik_map = self._ticker_to_cik()
        out: dict[str, dict] = {}
        for sym in self.symbols:
            cik = cik_map.get(str(sym).upper())
            if cik is None:
                self.warnings.append(f"no CIK for symbol {sym!r}")
                continue
            try:
                out[sym] = self._get_json(COMPANYFACTS_URL.format(cik=cik))
            except Exception as exc:  # keep going across the universe
                self.warnings.append(f"companyfacts fetch failed for {sym!r}: {exc}")
        return out

    # -------------------------------------------------------------- transform
    @staticmethod
    def _pick_concept(facts: dict, chain: list[tuple[str, str]]) -> list[dict]:
        """Flatten every unit array of the first present concept in the chain."""
        for tax, concept in chain:
            node = facts.get(tax, {}).get(concept)
            if not node:
                continue
            entries: list[dict] = []
            for arr in node.get("units", {}).values():
                entries.extend(arr)
            if entries:
                return entries
        return []

    @staticmethod
    def _dedup_vintages(df: pd.DataFrame) -> pd.DataFrame:
        """Keep the FIRST filing of each distinct value run per (instrument, field, period).

        Rows are ordered by ``filed`` within each (instrument_id, field, obs_date) group;
        a row is kept iff it is the first in its group or its value differs from the
        previously kept value. This collapses the endless re-reporting of an unchanged
        number to its original filing while preserving genuine restatements as later
        vintages (each at its own filing date). Restatements are never dropped.
        """
        if df.empty:
            return df
        df = df.sort_values(["instrument_id", "field", "obs_date", "filed"], kind="stable")
        keep: list[int] = []
        for _, grp in df.groupby(["instrument_id", "field", "obs_date"], sort=False):
            prev = object()  # sentinel distinct from any float value
            for idx, val in zip(grp.index, grp["value"]):
                if val != prev:
                    keep.append(idx)
                    prev = val
        return df.loc[keep]

    def transform(self, raw) -> pd.DataFrame:
        cols = ["obs_date", "instrument_id", "field", "value", "filed_ts", "asset_class"]
        rows = []
        for symbol, facts_json in (raw or {}).items():
            # Symbols arrive in the SEC/yfinance dash form (BRK-B); the master's plain
            # `symbol` is the dotted Wikipedia form, so try the yfinance vendor key too.
            iid = (self.resolve(symbol, self.vendor)
                   or self.resolve(symbol, "yfinance") or self.resolve(symbol))
            if iid is None:
                self.warnings.append(f"unresolved instrument for symbol {symbol!r}")
                continue
            facts = (facts_json or {}).get("facts", {})
            for field, chain in CONCEPTS.items():
                for e in self._pick_concept(facts, chain):
                    form = str(e.get("form", ""))
                    if not any(form.startswith(p) for p in _ALLOWED_FORM_PREFIXES):
                        continue
                    end, val, filed = e.get("end"), e.get("val"), e.get("filed")
                    if end is None or val is None or filed is None:
                        continue
                    # Duration concepts (eps/revenue) are reported under the SAME `end`
                    # for Q4 and the full fiscal year, often in the SAME filing — which
                    # collides on the (obs_date, field, available_from) key with two
                    # different values. Keep the quarterly duration; the annual figure
                    # is derivable and the quarterly one is what the signals consume.
                    start = e.get("start")
                    if start is not None:
                        try:
                            span = (pd.Timestamp(str(end))
                                    - pd.Timestamp(str(start))).days
                        except (TypeError, ValueError):
                            span = None
                        if span is not None and span > 200:
                            continue
                    try:
                        value = float(val)
                    except (TypeError, ValueError):
                        continue
                    rows.append({
                        "obs_date": pd.Timestamp(str(end)).normalize(),
                        "instrument_id": iid,
                        "field": field,
                        "value": value,
                        "filed": pd.Timestamp(str(filed)).normalize(),
                    })
        if not rows:
            return pd.DataFrame(columns=cols)

        df = pd.DataFrame(rows)
        df = self._dedup_vintages(df)
        # Same-day refilings that survive the vintage dedup (e.g. a 10-Q and its /A
        # amendment filed the same day with different values) share an available_from
        # and would trip the audit's duplicate check; the amendment (later in stable
        # sort order) wins deterministically.
        df = df.drop_duplicates(subset=["obs_date", "instrument_id", "field", "filed"],
                                keep="last")
        # available_from source: filing day @ 21:00 (localized to UTC by the "explicit"
        # availability rule). Kept tz-naive so the audit's numeric-column probe skips it.
        df["filed_ts"] = df["filed"] + pd.Timedelta(hours=21)
        df["asset_class"] = "equity"
        return (df[cols]
                .sort_values(["instrument_id", "field", "obs_date", "filed_ts"], kind="stable")
                .reset_index(drop=True))
