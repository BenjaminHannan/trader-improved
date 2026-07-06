"""Point-in-time correctness for `production.core.pit` (`asof_panel` / `asof_wide`).

The single rule under test: a row is visible at `as_of` iff `available_from <= as_of`
(boundary inclusive), and among visible vintages of the same
(obs_date, entity[, field]) the latest-available vintage wins. Everything downstream
in the system routes through here, so these are the corruption-guard tests.
"""
from __future__ import annotations

import pandas as pd

from production.core.pit import asof_panel, asof_wide

IID = "EQ:SYN00:2000-01-03"
SID = "DGS3MO_US"


def _row(obs, avail, value, ingested=None, entity=IID, entity_col="instrument_id",
         **extra) -> dict:
    obs = pd.Timestamp(obs)
    avail = pd.Timestamp(avail, tz="UTC")
    ingested = pd.Timestamp(ingested, tz="UTC") if ingested is not None else avail
    return {"obs_date": obs, entity_col: entity, "available_from": avail,
            "ingested_at": ingested, "source": "syn", "value": value, **extra}


def _frame(rows) -> pd.DataFrame:
    return pd.DataFrame(rows)


# --------------------------------------------------------------- boundary rule
def test_asof_boundary_is_inclusive_and_excludes_future():
    df = _frame([
        _row("2020-01-01", "2020-01-02 12:00", 1.0),  # available before as_of
        _row("2020-01-03", "2020-01-05 12:00", 2.0),  # available exactly at as_of
        _row("2020-01-06", "2020-01-07 12:00", 3.0),  # available after as_of
    ])
    as_of = pd.Timestamp("2020-01-05 12:00", tz="UTC")
    snap = asof_panel(df, as_of)
    got = set(snap["value"])
    assert 1.0 in got          # strictly before -> visible
    assert 2.0 in got          # exactly equal -> visible (inclusive boundary)
    assert 3.0 not in got      # after -> hidden


def test_asof_latest_visible_vintage_wins():
    # Two vintages of the SAME (obs_date, instrument_id); both visible at as_of.
    df = _frame([
        _row("2020-01-01", "2020-01-02 00:00", 10.0, ingested="2020-01-02 00:00"),
        _row("2020-01-01", "2020-01-04 00:00", 11.0, ingested="2020-01-04 00:00"),
    ])
    snap = asof_panel(df, pd.Timestamp("2020-01-10", tz="UTC"))
    assert len(snap) == 1
    assert snap["value"].iloc[0] == 11.0  # later available vintage wins


def test_asof_naive_timestamp_treated_as_utc():
    df = _frame([_row("2020-01-01", "2020-01-02 12:00", 1.0)])
    # Naive as_of exactly at the (UTC) availability instant must be inclusive.
    naive = asof_panel(df, pd.Timestamp("2020-01-02 12:00"))
    aware = asof_panel(df, pd.Timestamp("2020-01-02 12:00", tz="UTC"))
    assert len(naive) == 1
    assert len(aware) == 1
    # And just before the instant (naive, UTC-interpreted) nothing is visible.
    assert asof_panel(df, pd.Timestamp("2020-01-02 11:59")).empty


def test_asof_respects_field_in_dedup_key():
    # Same (obs_date, instrument_id) but different `field` -> distinct rows kept.
    df = _frame([
        _row("2020-01-01", "2020-01-02 00:00", 1.0, field="close"),
        _row("2020-01-01", "2020-01-02 00:00", 2.0, field="open"),
        # A later vintage of the close field only.
        _row("2020-01-01", "2020-01-03 00:00", 9.0, field="close"),
    ])
    snap = asof_panel(df, pd.Timestamp("2020-01-10", tz="UTC"))
    by_field = dict(zip(snap["field"], snap["value"]))
    assert by_field == {"close": 9.0, "open": 2.0}


def test_asof_series_id_panels_work():
    df = _frame([
        _row("2020-01-01", "2020-01-02 00:00", 5.0, entity=SID, entity_col="series_id"),
        _row("2020-01-02", "2020-01-09 00:00", 6.0, entity=SID, entity_col="series_id"),
    ])
    snap = asof_panel(df, pd.Timestamp("2020-01-05", tz="UTC"))
    assert list(snap["series_id"].unique()) == [SID]
    assert set(snap["value"]) == {5.0}  # the 2020-01-09 availability is still in the future


def test_asof_empty_when_as_of_predates_all_availability():
    df = _frame([_row("2020-01-01", "2020-01-02 00:00", 1.0)])
    assert asof_panel(df, pd.Timestamp("2019-12-31", tz="UTC")).empty


def test_asof_empty_input_passes_through():
    empty = _frame([_row("2020-01-01", "2020-01-02 00:00", 1.0)]).iloc[0:0]
    out = asof_panel(empty, pd.Timestamp("2020-01-10", tz="UTC"))
    assert out.empty


# ------------------------------------------------------------------ asof_wide
def test_asof_wide_pivots_and_respects_pit():
    # obs_date 2020-01-01 has an original value (100) and a later revision (150).
    # As of a date BEFORE the revision is available, the wide matrix must show the
    # original value, never the future revision.
    df = _frame([
        _row("2020-01-01", "2020-01-02 00:00", 100.0, ingested="2020-01-02 00:00"),
        _row("2020-01-01", "2020-01-20 00:00", 150.0, ingested="2020-01-20 00:00"),
        _row("2020-01-02", "2020-01-03 00:00", 200.0),
    ])
    wide = asof_wide(df, pd.Timestamp("2020-01-10", tz="UTC"))
    assert list(wide.columns) == [IID]
    assert list(wide.index) == [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02")]
    assert wide.loc[pd.Timestamp("2020-01-01"), IID] == 100.0  # pre-revision value
    assert wide.loc[pd.Timestamp("2020-01-02"), IID] == 200.0

    # After the revision is knowable, the revised value shows.
    wide_later = asof_wide(df, pd.Timestamp("2020-01-21", tz="UTC"))
    assert wide_later.loc[pd.Timestamp("2020-01-01"), IID] == 150.0
