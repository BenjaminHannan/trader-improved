"""Quarantine-list consumption in the backtest runner (wiki backlog #18).

``scripts/ingest.py --cross-check`` writes a return-space vendor-divergence report to
``<lake_root>/audit/cross_check_prices.json``; ``production.data.cross_check.quarantine_list``
turns it into instrument_ids too divergent between vendors to trust. This file tests
``scripts/run_backtest.py:apply_quarantine`` — the pure function that wires that list into
the backtest's traded universe (the ``instruments`` instrument_id->sleeve Series).

Class distinction under test (see apply_quarantine's docstring and
research/wiki/sources/etf-adjustment-methodology.md): quarantined EQUITY names are dropped
(ticker-reuse-class corruption, no exoneration); quarantined FX-ETF / commodity-ETF names
are KEPT with a loud warning (the divergence is a vendor adjusted-close *methodology*
artifact, not corrupt primary data, per the wiki source page's conclusion).

No network; no full backtest run — everything here is canned JSON + a pd.Series.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.run_backtest import apply_quarantine, quarantine_audit_path

EQ_BAD = "EQ:APC:1976-07-01"       # ticker-reuse-class equity, dropped
EQ_BAD2 = "EQ:MI:1976-07-01"       # second equity, dropped
EQ_OK = "EQ:AAPL:1980-12-12"       # equity NOT in the quarantine list, stays
FX_BAD = "FX:FXE:2005-12-01"       # fx-etf adjustment-artifact class, kept + warned
CO_BAD = "CO:CORN:2010-06-09"      # commodity-etf, same class, kept + warned
CR_OK = "CR:BTC:2017-01-01"        # untouched sleeve, not in quarantine list at all


def _report(flagged_ids: list[str], max_flag_frac: float = 0.05) -> dict:
    """A minimal cross_vendor_report-shaped dict whose quarantine_list() == flagged_ids.

    quarantine_list() thresholds on ``flag_frac > max_flag_frac`` (default arg 0.02), so
    every flagged id gets a flag_frac comfortably above that default threshold.
    """
    instruments = {
        iid: {
            "n_overlap": 200,
            "corr": 0.5,
            "max_abs_divergence_bp": 500.0,
            "flagged_dates": ["2024-01-02"],
            "flag_frac": max_flag_frac,
        }
        for iid in flagged_ids
    }
    return {
        "threshold_bp": 50.0,
        "min_overlap": 60,
        "instruments": instruments,
        "insufficient_overlap": {},
        "summary": {
            "instruments_checked": len(flagged_ids),
            "instruments_flagged": len(flagged_ids),
            "worst_offender": flagged_ids[0] if flagged_ids else None,
        },
    }


def _instruments() -> pd.Series:
    return pd.Series({
        EQ_BAD: "equity", EQ_BAD2: "equity", EQ_OK: "equity",
        FX_BAD: "fx_etf", CO_BAD: "commodity_etf", CR_OK: "crypto",
    })


def _write_audit(tmp_path, flagged_ids: list[str]):
    path = quarantine_audit_path(tmp_path / "data")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_report(flagged_ids)))
    return path


# --------------------------------------------------------- equity dropped, ETF kept+warned
def test_equity_dropped_etf_kept_with_warning(tmp_path, capsys):
    audit_path = _write_audit(tmp_path, [EQ_BAD, EQ_BAD2, FX_BAD, CO_BAD])
    out = apply_quarantine(_instruments(), audit_path)

    # equity names gone from the traded universe
    assert EQ_BAD not in out.index
    assert EQ_BAD2 not in out.index
    # untouched names (equity not flagged, and a sleeve absent from the list) survive
    assert EQ_OK in out.index
    assert CR_OK in out.index
    # ETF classes are KEPT despite being flagged
    assert FX_BAD in out.index
    assert CO_BAD in out.index
    assert out[FX_BAD] == "fx_etf"
    assert out[CO_BAD] == "commodity_etf"

    printed = capsys.readouterr().out
    assert "dropped 2 equity instrument(s)" in printed
    assert EQ_BAD in printed and EQ_BAD2 in printed
    assert "WARNING" in printed
    assert "2 non-equity instrument(s)" in printed
    assert "etf-adjustment-methodology.md" in printed


def test_equity_only_quarantine_drops_with_no_warning(tmp_path, capsys):
    audit_path = _write_audit(tmp_path, [EQ_BAD])
    out = apply_quarantine(_instruments(), audit_path)

    assert EQ_BAD not in out.index
    assert set(out.index) == set(_instruments().index) - {EQ_BAD}

    printed = capsys.readouterr().out
    assert "dropped 1 equity instrument(s)" in printed
    assert "WARNING" not in printed  # no non-equity flags -> no warn branch


def test_etf_only_quarantine_warns_with_no_drop(tmp_path, capsys):
    audit_path = _write_audit(tmp_path, [FX_BAD, CO_BAD])
    out = apply_quarantine(_instruments(), audit_path)

    # nothing dropped at all
    assert set(out.index) == set(_instruments().index)

    printed = capsys.readouterr().out
    assert "dropped" not in printed
    assert "WARNING" in printed
    assert FX_BAD in printed and CO_BAD in printed


# ------------------------------------------------------------------------- preview cap
def test_preview_truncates_long_lists(tmp_path, capsys):
    many_eq = [f"EQ:Z{i:02d}:2000-01-01" for i in range(8)]
    inst = pd.Series({iid: "equity" for iid in many_eq})
    audit_path = _write_audit(tmp_path, many_eq)
    out = apply_quarantine(inst, audit_path)

    assert out.empty
    printed = capsys.readouterr().out
    assert "dropped 8 equity instrument(s)" in printed
    assert "3 more" in printed  # 8 flagged, preview caps at 5


# --------------------------------------------------------------------------- no-op cases
def test_absent_audit_file_is_a_silent_noop(tmp_path, capsys):
    audit_path = quarantine_audit_path(tmp_path / "data")  # never written
    assert not audit_path.exists()

    original = _instruments()
    out = apply_quarantine(original, audit_path)

    pd.testing.assert_series_equal(out.sort_index(), original.sort_index())
    assert capsys.readouterr().out == ""  # no audit file -> identical behavior, no message


def test_no_ids_from_quarantine_present_in_universe_is_a_noop(tmp_path, capsys):
    # quarantine list contains only ids that are not in this run's instrument universe
    audit_path = _write_audit(tmp_path, ["EQ:NOTHERE:1999-01-01"])
    original = _instruments()
    out = apply_quarantine(original, audit_path)

    pd.testing.assert_series_equal(out.sort_index(), original.sort_index())
    assert capsys.readouterr().out == ""


def test_malformed_audit_file_does_not_sink_the_run(tmp_path, capsys):
    audit_path = quarantine_audit_path(tmp_path / "data")
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("{not valid json")

    original = _instruments()
    out = apply_quarantine(original, audit_path)

    pd.testing.assert_series_equal(out.sort_index(), original.sort_index())
    printed = capsys.readouterr().out
    assert "could not parse" in printed


# --------------------------------------------------------------------------- --no-quarantine
def test_no_quarantine_bypasses_filtering_with_loud_note(tmp_path, capsys):
    audit_path = _write_audit(tmp_path, [EQ_BAD, EQ_BAD2, FX_BAD])
    original = _instruments()
    out = apply_quarantine(original, audit_path, no_quarantine=True)

    # nothing dropped despite a populated quarantine list
    pd.testing.assert_series_equal(out.sort_index(), original.sort_index())

    printed = capsys.readouterr().out
    assert "--no-quarantine" in printed
    assert "BYPASSED" in printed


def test_no_quarantine_loud_note_even_when_audit_file_absent(tmp_path, capsys):
    audit_path = quarantine_audit_path(tmp_path / "data")  # never written
    out = apply_quarantine(_instruments(), audit_path, no_quarantine=True)

    assert set(out.index) == set(_instruments().index)
    printed = capsys.readouterr().out
    assert "--no-quarantine" in printed  # flag was used -> loud note regardless of file


# ------------------------------------------------------------------------- real-shape probe
def test_quarantine_audit_path_matches_lake_layout(tmp_path):
    p = quarantine_audit_path(tmp_path / "data")
    assert p == tmp_path / "data" / "audit" / "cross_check_prices.json"


def test_unknown_prefix_is_treated_as_non_equity_and_warned(tmp_path, capsys):
    """A malformed/unknown instrument_id prefix must never be silently dropped as equity."""
    weird = "ZZ:WEIRD:2000-01-01"
    inst = pd.Series({weird: "unknown_sleeve"})
    audit_path = _write_audit(tmp_path, [weird])
    out = apply_quarantine(inst, audit_path)

    assert weird in out.index  # kept, not dropped
    printed = capsys.readouterr().out
    assert "WARNING" in printed
    assert "dropped" not in printed
