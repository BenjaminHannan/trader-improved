"""Live IC monitoring — is a *deployed* factor still doing in production what it did in
training, and is it doing so without cheating?

This module is the production counterpart of ``production.alpha.ic``. There the rolling
IC feeds same-day sizing decisions and the point of the machinery is to *never* let an
unresolved forward-return window leak into a decision. Here we are no longer sizing; we
are watching a factor's realized IC drift over time. But the same embargo trap applies:
the live IC "as of" date ``t`` may only average IC observations whose ``horizon_days``
forward window has already resolved by ``t``. Mirror ``rolling_shrunk_ic`` exactly —
positional embargo on the IC index, not a calendar ``Timedelta`` cutoff (21 calendar
days < 21 trading days would let unresolved ICs leak in). The only difference from
``rolling_shrunk_ic`` is that we take a plain trailing *mean* rather than a shrunk mean:
a monitor wants the raw realized IC, not an estimator pulled toward zero.

Two failure modes get alarms:
- **decay**: the trailing live IC has collapsed toward zero relative to what training
  promised (``< ratio * train_ic`` on a same-sign basis);
- **sign flip**: the live IC has taken the *opposite* sign to training and stayed there
  for a full ``flip_window`` of observations — a factor that has inverted, not merely
  faded.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from production.alpha.ic import forward_returns, rank_ic
from production.core.config import REPO_ROOT


def _rolling_embargo_mean(ic_by_date: pd.Series, window: int,
                          horizon_days: int) -> pd.Series:
    """Trailing mean IC usable at each date, embargoing the unresolved forward window.

    Mirrors ``production.alpha.ic.rolling_shrunk_ic`` semantics exactly, but averages
    (rather than shrinks) the eligible tail. The value at position ``i`` (date ``t``)
    averages the up-to-``window`` most recent IC observations at positions ``j`` with
    ``j <= i - horizon_days`` — i.e. whose forward window has resolved by ``t``. The
    embargo is POSITIONAL on the IC index because ``horizon_days`` is a trading-day
    horizon and the index consists of trading dates; a calendar cutoff would
    under-embargo. Early dates with no eligible history are ``NaN``.
    """
    s = pd.Series(ic_by_date).dropna().sort_index()
    out: dict = {}
    for i, t in enumerate(s.index):
        j_max = i - horizon_days
        if j_max < 0:
            out[t] = float("nan")
            continue
        eligible = s.iloc[:j_max + 1]
        w = eligible.iloc[-window:]
        out[t] = float(w.mean())
    return pd.Series(out)


def live_rolling_ic(scores: pd.DataFrame, prices: pd.DataFrame,
                    sleeve_map: pd.Series, horizon_days: int,
                    window: int = 63) -> pd.DataFrame:
    """Trailing realized rank IC of a live score panel, per sleeve, embargo-safe.

    A thin wrapper over ``production.alpha.ic``: compute forward returns at the factor's
    horizon, score them against the (already z-scored) live signal via ``rank_ic``, then
    take a positional-embargo trailing mean of the per-date IC within each sleeve.

    Parameters
    ----------
    scores      : long panel ``[obs_date, instrument_id, value]`` — the live signal
                  scores actually used to size positions.
    prices      : curated price panel (needs ``obs_date, instrument_id, close``).
    sleeve_map  : ``pd.Series`` mapping ``instrument_id -> sleeve``.
    horizon_days: the factor's forward horizon (trading days). Sets both the forward
                  return window AND the embargo depth — they must match.
    window      : trailing IC observations to average (default 63, ~a quarter).

    Returns
    -------
    Long panel ``[obs_date, sleeve, live_ic]``. Dates with no resolved IC history yet
    carry ``NaN`` live IC (the embargo warm-up), kept so the caller sees the full span.
    """
    fwd = forward_returns(prices, horizon_days)
    ic = rank_ic(scores, fwd, sleeve_map)  # [obs_date, sleeve, rank_ic, n_names]

    rows: list[tuple] = []
    for sleeve, g in ic.groupby("sleeve", sort=False):
        by_date = g.set_index("obs_date")["rank_ic"].sort_index()
        roll = _rolling_embargo_mean(by_date, window=window, horizon_days=horizon_days)
        for d, v in roll.items():
            rows.append((d, sleeve, v))
    return pd.DataFrame(rows, columns=["obs_date", "sleeve", "live_ic"])


def decay_alarms(live_ic: pd.Series, train_ic: float, factor: str,
                 ratio: float = 0.25, flip_window: int = 63) -> list[dict]:
    """Raise decay / sign-flip alarms for one factor from its live IC series.

    ``live_ic`` is the trailing embargo-safe IC series (e.g. one sleeve column of
    ``live_rolling_ic``). ``train_ic`` is the IC the factor showed on the training span.
    Comparisons are on a *same-sign basis*: everything is aligned by the sign of
    ``train_ic`` so that a healthy factor has positive aligned IC ≈ ``|train_ic|``.

    - **decay** fires when the latest trailing live IC has fallen below
      ``ratio * train_ic`` (same-sign) — the edge has collapsed toward (or through) zero.
    - **sign_flip** fires when the aligned live IC has been negative for the entire last
      ``flip_window`` observations — a persistent inversion, not a one-off wobble.

    Returns a list of alarm dicts (possibly empty). Each dict carries the factor, alarm
    ``kind``, the numbers behind the trip, and the ``as_of`` date.
    """
    s = pd.Series(live_ic).dropna().sort_index()
    alarms: list[dict] = []
    if s.empty:
        return alarms

    sign = 1.0 if train_ic >= 0 else -1.0
    aligned = s * sign                       # healthy => positive, ~|train_ic|
    threshold = ratio * abs(float(train_ic))
    as_of = s.index[-1]
    latest_aligned = float(aligned.iloc[-1])
    latest_live = float(s.iloc[-1])

    if latest_aligned < threshold:
        alarms.append({
            "factor": factor,
            "kind": "decay",
            "as_of": str(pd.Timestamp(as_of).date()),
            "train_ic": float(train_ic),
            "live_ic": latest_live,
            "threshold": float(threshold),
            "ratio": float(ratio),
        })

    tail = aligned.iloc[-flip_window:]
    if len(tail) >= flip_window and bool((tail < 0).all()):
        alarms.append({
            "factor": factor,
            "kind": "sign_flip",
            "as_of": str(pd.Timestamp(as_of).date()),
            "train_ic": float(train_ic),
            "live_ic": latest_live,
            "flip_window": int(flip_window),
        })

    return alarms


def monitor_report(alarms: list[dict],
                   live_vs_train: pd.DataFrame | list[dict] | dict) -> dict:
    """Assemble the monitor snapshot: which factors are in alarm, plus the live-vs-train
    IC table for every monitored factor.

    ``alarms`` is the concatenation of every factor's ``decay_alarms`` output.
    ``live_vs_train`` is a per-factor table (DataFrame, list of record dicts, or a
    ``{factor: {...}}`` mapping) of at least ``train_ic`` and ``live_ic``. The returned
    dict is JSON-serializable via ``write_monitor_report``.
    """
    table = _normalize_table(live_vs_train)
    factors_in_alarm = sorted({a.get("factor") for a in alarms if a.get("factor")})
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_factors": len(table),
        "n_alarms": len(alarms),
        "factors_in_alarm": factors_in_alarm,
        "alarms": list(alarms),
        "live_vs_train": table,
    }


def _normalize_table(live_vs_train) -> list[dict]:
    """Coerce the live-vs-train table into a list of plain record dicts."""
    if isinstance(live_vs_train, pd.DataFrame):
        return live_vs_train.to_dict(orient="records")
    if isinstance(live_vs_train, dict):
        rows = []
        for factor, rec in live_vs_train.items():
            row = {"factor": factor}
            row.update(rec if isinstance(rec, dict) else {"value": rec})
            rows.append(row)
        return rows
    return [dict(r) for r in live_vs_train]


def _coerce(o):
    """Make numpy / pandas scalars JSON-friendly (NaN/inf -> None). Mirrors report.py."""
    if isinstance(o, dict):
        return {str(k): _coerce(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_coerce(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if not np.isfinite(f) else f
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp, datetime)):
        return str(o)
    return o


def write_monitor_report(report: dict, out_dir: str | Path = "reports") -> Path:
    """Write ``monitor_<UTCtimestamp>.json`` into ``reports/`` (never ``data/``).

    Returns the JSON path. Mirrors ``production.backtest.report.write_report``: relative
    ``out_dir`` is resolved under ``REPO_ROOT``; values are coerced so ``NaN``/``inf``
    never leak into the file (``allow_nan=False``).
    """
    out = Path(out_dir)
    if not out.is_absolute():
        out = REPO_ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out / f"monitor_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(_coerce(report), f, indent=2, allow_nan=False)
    return json_path
