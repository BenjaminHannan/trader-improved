"""Point-in-time universe membership.

Three membership regimes, one per sleeve family:

* **equity (S&P 500)** — reconstructed by walking the Wikipedia "Selected changes"
  table *backwards* from today's constituent list. Survivorship bias is the classic
  killer here: backtesting on today's members only would silently drop every name that
  was ever removed. The walk-back rebuilds the historical membership intervals.
* **crypto** — a monthly, trailing-30d dollar-volume ranked top-N, with a listing-age
  floor. The ranking uses volume *shifted one day* so a same-day volume spike can never
  pull a name into that very refresh (no lookahead).
* **fx / commodity ETFs** — static lists keyed on the instrument master's inception
  dates.

Membership frames are long: ``[<key>, universe, effective_from, effective_to]`` where
``effective_to`` NaT means "still current" and the key is ``symbol`` (equity layer,
pre-instrument-mint) or ``instrument_id`` (crypto/static).
"""
from __future__ import annotations

import pandas as pd

from production.core.config import universe_config
from production.reference.instruments import build_instrument_master

WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_UA = "trader-improved/1.0 (research; contact via repo)"


# --------------------------------------------------------------- HTML parsing
def _read_tables(html: str) -> list[pd.DataFrame]:
    from io import StringIO

    return pd.read_html(StringIO(html))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse a MultiIndex header (Wikipedia's Added/Removed -> Ticker/Security
    two-row header) into space-joined strings, dropping ``Unnamed`` filler."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [
            " ".join(
                str(x) for x in tup
                if str(x) and not str(x).startswith("Unnamed")
            ).strip()
            for tup in df.columns
        ]
    else:
        df = df.copy()
        df.columns = [str(c) for c in df.columns]
    return df


def _find_col(df: pd.DataFrame, *needles: str) -> str | None:
    """First column whose lowercased name contains every needle."""
    for c in df.columns:
        cl = str(c).lower()
        if all(n.lower() in cl for n in needles):
            return c
    return None


def _is_blank(val) -> bool:
    """A ticker cell is blank if it is None, NaN, or empty after stripping."""
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    return str(val).strip() == "" or str(val).strip().lower() == "nan"


def _clean_ticker(val) -> str | None:
    """Normalise a ticker cell; blank/NaN -> None."""
    if _is_blank(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    # Wikipedia occasionally footnotes tickers; keep the leading token.
    return s.split()[0].replace("\xa0", "")


def _parse_current(html: str) -> list[str]:
    tables = _read_tables(html)
    df = _flatten_columns(tables[0])
    sym_col = _find_col(df, "symbol") or _find_col(df, "ticker") or df.columns[0]
    return [t for t in (_clean_ticker(v) for v in df[sym_col]) if t]


def sector_map_from_wikipedia(html_current: str | None = None) -> dict[str, str]:
    """Parse ``symbol -> GICS sector`` from the current-constituents table.

    PIT-honest caveat: this is a *current* snapshot of each name's sector — the same
    treatment ``scripts/run_backtest.py:_sector_map`` already documents. Delisted names
    absent from today's table simply get no sector (their master row stays ``None``).

    Pass `html_current` to inject page HTML (tests do this — no network in pytest); when
    ``None`` the S&P 500 page is fetched once with a UA header.
    """
    if html_current is None:
        import requests

        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": _UA}, timeout=30)
        resp.raise_for_status()
        html_current = resp.text

    df = _flatten_columns(_read_tables(html_current)[0])
    sym_col = _find_col(df, "symbol") or _find_col(df, "ticker") or df.columns[0]
    sec_col = _find_col(df, "gics", "sector") or _find_col(df, "sector")
    if sec_col is None:
        return {}
    out: dict[str, str] = {}
    for sym, sec in zip(df[sym_col], df[sec_col]):
        ticker = _clean_ticker(sym)
        if ticker and not _is_blank(sec):
            out[ticker] = str(sec).strip()
    return out


def _parse_changes(html: str) -> pd.DataFrame:
    tables = _read_tables(html)
    # The changes table is the one carrying Added/Removed ticker columns.
    chosen = None
    for t in tables:
        f = _flatten_columns(t)
        if _find_col(f, "added", "ticker") and _find_col(f, "removed", "ticker"):
            chosen = f
            break
    if chosen is None:
        chosen = _flatten_columns(tables[-1])

    date_col = _find_col(chosen, "date") or chosen.columns[0]
    add_col = _find_col(chosen, "added", "ticker")
    rem_col = _find_col(chosen, "removed", "ticker")
    out = pd.DataFrame({
        "date": pd.to_datetime(chosen[date_col], errors="coerce"),
        "added": [_clean_ticker(v) for v in chosen[add_col]] if add_col else None,
        "removed": [_clean_ticker(v) for v in chosen[rem_col]] if rem_col else None,
    })
    return out.dropna(subset=["date"]).reset_index(drop=True)


# ----------------------------------------------------------- S&P 500 walk-back
def sp500_membership_from_wikipedia(html_current: str | None = None,
                                    html_changes: str | None = None) -> pd.DataFrame:
    """Reconstruct symbol-level S&P 500 membership intervals from Wikipedia.

    Pass `html_current` / `html_changes` to inject page HTML (tests do this — no
    network in pytest). When both are ``None`` the current page is fetched once with a
    UA header and both tables are read from it.

    Returns long ``[symbol, universe="sp500", effective_from, effective_to]``. NaT
    ``effective_from`` means "member since before the change history begins"; NaT
    ``effective_to`` means "still a member". A change row with a blank added *or*
    removed cell (a pure add or pure removal) is handled.

    Note: the pinned contract lists ``instrument_id`` as the key column. Membership at
    this layer is intrinsically symbol-keyed (ids are minted downstream from these very
    first-seen dates); use :func:`to_instrument_membership` to convert once the master
    exists.
    """
    if html_current is None and html_changes is None:
        import requests

        resp = requests.get(WIKI_SP500_URL, headers={"User-Agent": _UA}, timeout=30)
        resp.raise_for_status()
        html_current = html_changes = resp.text

    current = _parse_current(html_current)
    changes = _parse_changes(html_changes)

    # active: symbol -> effective_to of the currently-open (going-backwards) interval.
    # Current constituents are open at the NaT (still-current) end.
    active: dict[str, pd.Timestamp] = {sym: pd.NaT for sym in current}
    intervals: list[dict] = []

    # Process newest change first: each event tells us the state *before* its date.
    for ev in changes.sort_values("date", ascending=False).itertuples(index=False):
        date = ev.date
        added = None if _is_blank(ev.added) else ev.added
        removed = None if _is_blank(ev.removed) else ev.removed

        if added is not None:
            # `added` joined on `date`; before `date` it was not a member, so its
            # currently-open interval starts here and is finalised.
            if added in active:
                eff_to = active.pop(added)
                intervals.append({"symbol": added, "effective_from": date,
                                  "effective_to": eff_to})
            else:
                # Added then later removed, but we never saw the removal (partial
                # change history). Record the single knowable interval starting here.
                intervals.append({"symbol": added, "effective_from": date,
                                  "effective_to": pd.NaT})

        if removed is not None:
            # `removed` left on `date`; before `date` it WAS a member. Open a
            # backward interval ending here (its start is set by an earlier change or
            # left NaT = "since before history").
            if removed in active:
                # Already open (re-added earlier in real time): close that segment
                # with an unknown start before reopening the older one.
                eff_to = active.pop(removed)
                intervals.append({"symbol": removed, "effective_from": pd.NaT,
                                  "effective_to": eff_to})
            active[removed] = date

    for sym, eff_to in active.items():
        intervals.append({"symbol": sym, "effective_from": pd.NaT,
                          "effective_to": eff_to})

    out = pd.DataFrame(intervals, columns=["symbol", "effective_from", "effective_to"])
    out.insert(1, "universe", "sp500")
    return out.sort_values(["symbol", "effective_from"]).reset_index(drop=True)


def to_instrument_membership(membership: pd.DataFrame,
                             master: pd.DataFrame) -> pd.DataFrame:
    """Convert a symbol-keyed membership frame to instrument-id keyed via the master.

    A symbol maps to the equity instrument whose validity window overlaps the
    membership interval (in practice one equity row per symbol). Symbols absent from
    the master are dropped with their intervals.
    """
    eq = master[master["sleeve"] == "equity"]
    sym_to_id = {}
    for row in eq.itertuples(index=False):
        # Keep the earliest-listed id per symbol (stable synthetic key).
        sym_to_id.setdefault(row.symbol, row.instrument_id)

    m = membership[membership["symbol"].isin(sym_to_id)].copy()
    m["instrument_id"] = m["symbol"].map(sym_to_id)
    cols = ["instrument_id", "universe", "effective_from", "effective_to"]
    return m[cols].reset_index(drop=True)


# ------------------------------------------------------------ crypto top-N
def _refresh_dates(dates: pd.Series) -> list[pd.Timestamp]:
    """First observed date of each calendar month — the monthly refresh grid."""
    d = pd.to_datetime(pd.Series(sorted(pd.unique(dates))))
    firsts = d.groupby([d.dt.year, d.dt.month]).min()
    return list(pd.to_datetime(firsts.values))


def crypto_membership(prices: pd.DataFrame, top_n: int = 25,
                      min_listing_days: int = 90) -> pd.DataFrame:
    """Monthly top-N crypto universe by trailing-30d dollar volume.

    At each month-start refresh ``R``:

    * eligibility requires the instrument to have been listed >= `min_listing_days`
      (``R - first_obs``);
    * the ranking statistic is the mean ``dollar_volume`` over the trailing 30 days
      **strictly before R** (``obs_date < R``) — i.e. shifted one day, so a spike on
      the refresh day itself cannot pull a name in;
    * the top `top_n` eligible names are members from ``R`` until the next refresh.

    Returns long ``[instrument_id, universe="crypto", effective_from, effective_to]``
    with contiguous memberships merged into single intervals.
    """
    if prices.empty:
        return pd.DataFrame(
            columns=["instrument_id", "universe", "effective_from", "effective_to"])

    px = prices[["obs_date", "instrument_id", "dollar_volume"]].copy()
    px["obs_date"] = pd.to_datetime(px["obs_date"])
    first_obs = px.groupby("instrument_id")["obs_date"].min()

    refreshes = _refresh_dates(px["obs_date"])
    selected: dict[pd.Timestamp, set[str]] = {}
    for r in refreshes:
        window = px[(px["obs_date"] < r)
                    & (px["obs_date"] >= r - pd.Timedelta(days=30))]
        if window.empty:
            selected[r] = set()
            continue
        adv = window.groupby("instrument_id")["dollar_volume"].mean()
        # listing-age floor
        age_days = (r - first_obs).dt.days
        eligible = adv[adv.index.map(lambda i: age_days.get(i, -1) >= min_listing_days)]
        top = eligible.sort_values(ascending=False).head(top_n)
        selected[r] = set(top.index)

    # Build contiguous intervals per instrument from the per-refresh selections.
    next_refresh = {refreshes[i]: (refreshes[i + 1] if i + 1 < len(refreshes) else pd.NaT)
                    for i in range(len(refreshes))}
    rows: list[dict] = []
    all_ids = sorted(px["instrument_id"].unique())
    for iid in all_ids:
        run_start = None
        run_end = None
        for r in refreshes:
            if iid in selected[r]:
                if run_start is None:
                    run_start = r
                run_end = next_refresh[r]
            else:
                if run_start is not None:
                    rows.append({"instrument_id": iid, "effective_from": run_start,
                                "effective_to": run_end})
                    run_start = None
        if run_start is not None:
            rows.append({"instrument_id": iid, "effective_from": run_start,
                        "effective_to": run_end})

    out = pd.DataFrame(rows, columns=["instrument_id", "effective_from", "effective_to"])
    out.insert(1, "universe", "crypto")
    return out.sort_values(["instrument_id", "effective_from"]).reset_index(drop=True)


# --------------------------------------------------------------- static ETFs
def static_membership(sleeve: str) -> pd.DataFrame:
    """Static membership for a fixed-list sleeve from instrument-master inception dates.

    Each instrument is a member from its ``valid_from`` (inception) onward, forever
    (``effective_to`` NaT). `universe` is the sleeve name.
    """
    master = build_instrument_master()
    sub = master[master["sleeve"] == sleeve]
    out = pd.DataFrame({
        "instrument_id": sub["instrument_id"].values,
        "universe": sleeve,
        "effective_from": pd.to_datetime(sub["valid_from"].values),
        "effective_to": pd.NaT,
    })
    return out.sort_values("instrument_id").reset_index(drop=True)


# ------------------------------------------------------------- as-of query
def membership_as_of(membership: pd.DataFrame, universe: str, date) -> list[str]:
    """Sorted keys active in `universe` on `date`.

    Active means ``effective_from <= date`` (NaT from = since before history) and
    ``date < effective_to`` (NaT to = still current). The removal date is the first
    day a name is *not* a member — membership is the half-open interval
    ``[effective_from, effective_to)``.
    """
    date = pd.Timestamp(date)
    m = membership[membership["universe"] == universe]
    key = "instrument_id" if "instrument_id" in m.columns else "symbol"
    ef = pd.to_datetime(m["effective_from"])
    et = pd.to_datetime(m["effective_to"])
    active = ((ef.isna() | (ef <= date)) & (et.isna() | (date < et)))
    return sorted(m.loc[active, key].unique())
