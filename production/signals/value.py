"""Value signals: long-horizon reversal (equity) and crypto market-cap/TVL value.

`lt_reversal_5y` (De Bondt-Thaler long-term reversal) is registered in code as a
candidate but intentionally NOT listed in ``configs/factors.yaml`` — it needs ~5y of
history and stays a code-level candidate until it has been through the gate. It is
covered by the corruption harness like any registered signal.

`mcap_tvl` (CryptoMcapTvl) is the crypto "value" factor: a cheap protocol earns fee
revenue proportional to the capital locked in it (TVL) relative to its market cap, so
``log(TVL / mcap)`` is a fundamentals-vs-price ratio — high means the market pays little
per unit of locked capital. Market-cap and TVL are lag-stamped snapshot sources
(CoinGecko / DefiLlama), pulled the morning after their observation date, so — exactly
like the macro carry join — the value at date D may use only snapshot rows with
``available_from <= end of day D``. That availability-date ``merge_asof`` is what keeps
it point-in-time. Assets with no TVL (BTC and other non-smart-contract chains) carry no
TVL row and drop out of the factor naturally.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from production.signals.base import OUTPUT_COLUMNS, Signal, register


@register
class LongHorizonReversal(Signal):
    """Long-term reversal: ``-(1260d return, skipping the most recent 252d)``.

    ``-(close.shift(252) / close.shift(1260) - 1)`` — the 5y-ago-to-1y-ago return,
    negated: multi-year winners tend to underperform subsequently. The recent 12
    months are skipped so this does not collide with 12-1 momentum.

    Equity-only for now; a crypto market-cap/TVL value variant is deferred to Stage 2.
    """

    name = "lt_reversal_5y"
    sleeves = ["equity"]
    required_datasets = ["prices"]
    min_history_days = 1300
    horizon_days = 63

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        p = self._restrict_to_sleeves(data["prices"])
        p = p.sort_values(["instrument_id", "obs_date"], kind="stable").reset_index(drop=True)
        if p.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        close = p.groupby("instrument_id", sort=False)["close"]
        value = -(close.shift(252) / close.shift(1260) - 1.0)
        out = pd.DataFrame({"obs_date": p["obs_date"], "instrument_id": p["instrument_id"],
                            "value": value})
        return self._finalize(out, anchor=p)


def _avail_step_by_instrument(df: pd.DataFrame, value_col: str) -> dict[str, pd.DataFrame]:
    """Reduce a per-instrument snapshot frame to per-instrument availability step functions.

    Each snapshot row's ``available_from`` (a UTC timestamp) is normalized to its date:
    the value is usable at any decision date D once ``available_from <= end of day D``,
    i.e. once D reaches that availability date. Ties on the same availability date keep
    the latest-arriving vintage. Returns ``{instrument_id: [avail_date, value_col]}``
    with each frame sorted by ``avail_date`` (merge_asof-ready).
    """
    if df is None or df.empty:
        return {}
    tmp = df[["instrument_id", "available_from", value_col]].copy()
    avail = pd.to_datetime(tmp["available_from"], utc=True)
    tmp["avail_date"] = avail.dt.normalize().dt.tz_localize(None)
    tmp["_af"] = avail
    tmp = (tmp.sort_values(["instrument_id", "avail_date", "_af"], kind="stable")
           .drop_duplicates(["instrument_id", "avail_date"], keep="last"))
    out: dict[str, pd.DataFrame] = {}
    for iid, grp in tmp.groupby("instrument_id", sort=False):
        out[iid] = grp[["avail_date", value_col]].reset_index(drop=True)
    return out


def _eps_vintages(fundamentals: pd.DataFrame, iid: str) -> pd.DataFrame:
    """Per-instrument eps vintage rows ``[obs_date(period end), avail_date, value]``.

    ``avail_date`` is each filing's ``available_from`` normalized to its UTC date: the
    value is usable at decision date D once ``available_from <= end of day D``. Rows are
    sorted by (avail_date, available_from) so a later-arriving vintage of a period
    overrides an earlier one from its availability date onward — exactly the PIT vintage
    semantics of the fundamentals lake.
    """
    e = fundamentals[(fundamentals["instrument_id"] == iid)
                     & (fundamentals["field"] == "eps")]
    if e.empty:
        return pd.DataFrame(columns=["obs_date", "avail_date", "value"])
    af = pd.to_datetime(e["available_from"], utc=True)
    out = pd.DataFrame({
        "obs_date": pd.to_datetime(e["obs_date"]).to_numpy(),
        "avail_date": af.dt.normalize().dt.tz_localize(None).to_numpy(),
        "_af": af.to_numpy(),
        "value": e["value"].to_numpy(),
    })
    return out.sort_values(["avail_date", "_af"], kind="stable").reset_index(drop=True)


@register
class EarningsYield(Signal):
    """Trailing-twelve-month diluted EPS over price: ``ttm_eps(filed by D) / close(D)``.

    A classic value factor. At each price date D the numerator is the sum of the four
    most recent fiscal quarters' EPS whose 10-Q/10-K was *filed on or before D* (using
    each period's latest vintage knowable by D); the denominator is the close at D. A
    quarter whose filing lands AFTER D contributes nothing to D — that filing-date gate,
    not the period-end, is what keeps the factor point-in-time. Emitted on price dates.
    """

    name = "earnings_yield"
    sleeves = ["equity"]
    required_datasets = ["prices", "fundamentals"]
    min_history_days = 63
    horizon_days = 21

    @staticmethod
    def _ttm(ev: pd.DataFrame, dates) -> "list":
        """Trailing-4-quarter EPS sum at each (sorted) decision date.

        Walks the availability-sorted vintages once, keeping the latest value per period
        knowable so far; at each date the sum of the four most recent periods (by period
        end) is returned, or NaN if fewer than four quarters have been filed.
        """
        period = ev["obs_date"].to_numpy()
        avail = ev["avail_date"].to_numpy()
        val = ev["value"].to_numpy()
        n = len(ev)
        res = [float("nan")] * len(dates)
        view: dict = {}
        i = 0
        for di, D in enumerate(dates):
            while i < n and avail[i] <= D:
                view[period[i]] = val[i]  # later avail overrides -> latest vintage wins
                i += 1
            if len(view) >= 4:
                recent = sorted(view)[-4:]
                res[di] = float(sum(view[p] for p in recent))
        return res

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._restrict_to_sleeves(data["prices"])
        fundamentals = data.get("fundamentals")
        if prices.empty or fundamentals is None or fundamentals.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        frames = []
        for iid, grp in prices.groupby("instrument_id", sort=False):
            ev = _eps_vintages(fundamentals, iid)
            if ev.empty:
                continue
            grp = grp.sort_values("obs_date", kind="stable")
            dates = pd.to_datetime(grp["obs_date"]).to_numpy()
            ttm = self._ttm(ev, dates)
            close = pd.to_numeric(grp["close"], errors="coerce").to_numpy()
            value = np.where(close > 0, np.asarray(ttm, dtype=float) / close, np.nan)
            frames.append(pd.DataFrame({"obs_date": grp["obs_date"].to_numpy(),
                                        "instrument_id": iid, "value": value}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)


@register
class PostEarningsDrift(Signal):
    """Post-earnings-announcement drift (PEAD): standardized earnings surprise, decaying.

    Bernard & Thomas (1989): stock returns drift in the direction of an earnings
    surprise for weeks after the announcement. The surprise here is a seasonal (year-
    over-year) EPS change scaled by its own magnitude::

        surprise = (eps_latest_quarter - eps_4_quarters_prior)
                   / max(|eps_4_quarters_prior|, 0.25)

    (the 0.25 floor tames the divide-by-near-zero when the year-ago quarter was around
    breakeven). The surprise ``prints`` at the announcement — the first date the quarter
    is *filed* — and is emitted, held constant, for 63 trading days of drift, then goes
    absent. It is never emitted before the filing date, and the year-ago comparison uses
    only the vintage knowable at the announcement, so the signal is point-in-time.
    """

    name = "pead"
    sleeves = ["equity"]
    required_datasets = ["prices", "fundamentals"]
    min_history_days = 63
    horizon_days = 21

    DRIFT_TRADING_DAYS = 63

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._restrict_to_sleeves(data["prices"])
        fundamentals = data.get("fundamentals")
        if prices.empty or fundamentals is None or fundamentals.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)

        frames = []
        for iid, grp in prices.groupby("instrument_id", sort=False):
            ev = _eps_vintages(fundamentals, iid)
            if ev.empty:
                continue
            events = self._surprise_events(ev)
            if not events:
                continue
            grp = grp.sort_values("obs_date", kind="stable")
            pdates = pd.to_datetime(grp["obs_date"]).to_numpy()
            value = np.full(len(pdates), np.nan)
            # emergence -> first price index on/after the filing date; assign a constant
            # surprise over the next DRIFT_TRADING_DAYS price dates. Events sorted
            # ascending so a fresh announcement overrides the tail of an older drift.
            for emergence, surprise in events:
                start = int(np.searchsorted(pdates, emergence, side="left"))
                if start >= len(pdates):
                    continue
                end = min(start + self.DRIFT_TRADING_DAYS, len(pdates))
                value[start:end] = surprise
            frames.append(pd.DataFrame({"obs_date": grp["obs_date"].to_numpy(),
                                        "instrument_id": iid, "value": value}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)

    @staticmethod
    def _surprise_events(ev: pd.DataFrame) -> list[tuple]:
        """List of ``(emergence_date, surprise)`` — one per quarterly announcement.

        Each period's *emergence* is the first date it was filed (min avail_date). The
        year-ago comparison quarter is the period four back by period end; its value is
        the latest vintage knowable at the announcement (avail_date <= emergence), so no
        future restatement leaks into the surprise.
        """
        period = ev["obs_date"].to_numpy()
        avail = ev["avail_date"].to_numpy()
        val = ev["value"].to_numpy()
        n = len(ev)

        # First-filing (emergence) date and first-filed value per period.
        emergence: dict = {}
        first_val: dict = {}
        for p, a, v in zip(period, avail, val):
            if p not in emergence or a < emergence[p]:
                emergence[p] = a
                first_val[p] = v
            elif a == emergence[p] and p not in first_val:
                first_val[p] = v
        ordered_periods = sorted(emergence)

        events = []
        for k, P in enumerate(ordered_periods):
            if k < 4:
                continue
            A = emergence[P]
            P4 = ordered_periods[k - 4]
            # value of the year-ago quarter as latest vintage knowable at A.
            prior = None
            for i in range(n):
                if avail[i] <= A and period[i] == P4:
                    prior = val[i]  # avail-sorted -> last write is latest vintage <= A
            if prior is None:
                continue
            surprise = (first_val[P] - prior) / max(abs(prior), 0.25)
            events.append((A, float(surprise)))
        events.sort(key=lambda t: t[0])
        return events


@register
class CryptoMcapTvl(Signal):
    """Crypto value: ``log(TVL / mcap)`` per crypto instrument.

    Total value locked (DefiLlama) over market cap (CoinGecko): a fundamentals-to-price
    ratio. Both are lag-stamped snapshot sources joined onto each instrument's own price
    dates by *availability date* (backward ``merge_asof``), so the value at date D uses
    only snapshot rows knowable by end of day D. Instruments without a TVL series (e.g.
    BTC, a non-smart-contract chain) produce no ratio and drop out of the factor.
    """

    name = "mcap_tvl"
    sleeves = ["crypto"]
    required_datasets = ["prices", "mcap", "tvl"]
    min_history_days = 30
    horizon_days = 21

    def compute(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        prices = self._restrict_to_sleeves(data["prices"])
        if prices.empty:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        mcap_avail = _avail_step_by_instrument(data["mcap"], "mcap")
        tvl_avail = _avail_step_by_instrument(data["tvl"], "tvl")

        frames = []
        for iid, grp in prices.groupby("instrument_id", sort=False):
            mc = mcap_avail.get(iid)
            tv = tvl_avail.get(iid)
            if mc is None or tv is None or mc.empty or tv.empty:
                continue  # no TVL (e.g. BTC) -> instrument drops out of the factor
            dates = (grp[["obs_date"]].sort_values("obs_date", kind="stable")
                     .reset_index(drop=True))
            m = pd.merge_asof(dates, mc, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            t = pd.merge_asof(dates, tv, left_on="obs_date", right_on="avail_date",
                              direction="backward")
            ratio = t["tvl"] / m["mcap"]
            value = np.log(ratio.where(ratio > 0))  # non-positive/NaN -> NaN, dropped
            frames.append(pd.DataFrame({"obs_date": dates["obs_date"],
                                        "instrument_id": iid, "value": value}))
        if not frames:
            return pd.DataFrame(columns=OUTPUT_COLUMNS)
        out = pd.concat(frames, ignore_index=True)
        return self._finalize(out, anchor=prices)
