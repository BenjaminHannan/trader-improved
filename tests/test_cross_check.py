"""Cross-vendor price cross-check tests — NO NETWORK.

Synthetic curated price frames are built from the shared GBM maker and relabelled with
two vendor `source` tags (yfinance vs stooq). The cross-check compares them in RETURN
space; these tests pin the properties that make it useful: identical feeds are silent,
a split mis-applied in one feed flags exactly the corrupt date, missing bars do not
false-positive, thin overlap is set aside, and the CLI runs end to end on a seeded lake.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from production.core.lake import Lake
from production.data.cross_check import (
    cross_vendor_report, quarantine_list, write_cross_check_audit,
)
from tests.conftest import make_gbm_prices

import scripts.ingest as ingest

AAA = "EQ:AAA:2000-01-03"
BBB = "EQ:BBB:2000-01-03"


def _feeds(source_p: str = "yfinance", source_s: str = "stooq"):
    """Two identical curated price feeds differing only in their `source` tag."""
    base = make_gbm_prices({AAA: "equity", BBB: "equity"},
                           start="2020-01-02", end="2020-12-31")
    return base.copy().assign(source=source_p), base.copy().assign(source=source_s)


def _dates(df: pd.DataFrame, iid: str) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(df.loc[df["instrument_id"] == iid, "obs_date"].unique()))


# ------------------------------------------------------------- (1) identical -> silent
def test_identical_feeds_flag_nothing():
    report = cross_vendor_report(*_feeds())
    assert report["summary"]["instruments_checked"] == 2
    assert report["summary"]["instruments_flagged"] == 0
    for iid in (AAA, BBB):
        r = report["instruments"][iid]
        assert r["flagged_dates"] == []
        assert r["max_abs_divergence_bp"] == pytest.approx(0.0)
        assert r["corr"] == pytest.approx(1.0)
    assert quarantine_list(report) == []


# ------------------------------------------------- (2) planted 2:1 split -> one bad day
def test_planted_split_flags_exactly_the_corrupt_date():
    primary, secondary = _feeds()
    D = _dates(secondary, AAA)[100]
    bad = (secondary["instrument_id"] == AAA) & (secondary["obs_date"] >= D)
    secondary.loc[bad, "close"] = secondary.loc[bad, "close"] * 0.5  # 2:1 split in ONE feed

    report = cross_vendor_report(primary, secondary)

    # The divergence is a single day in return space: only date D is flagged.
    assert report["instruments"][AAA]["flagged_dates"] == [D.strftime("%Y-%m-%d")]
    assert report["instruments"][BBB]["flagged_dates"] == []
    assert report["summary"]["instruments_flagged"] == 1
    assert report["summary"]["worst_offender"] == AAA
    # Big return-space divergence (~5000bp), far above the 50bp floor.
    assert report["instruments"][AAA]["max_abs_divergence_bp"] > 1000

    # Below the default fraction (one bad day out of a year) it is not quarantined,
    # but a strict threshold catches it.
    assert AAA not in quarantine_list(report)
    strict = quarantine_list(report, max_flag_frac=0.0)
    assert strict == [AAA]


# --------------------------------------------------- (3) missing bars -> no false alarm
def test_missing_dates_do_not_false_positive():
    primary, secondary = _feeds()
    gap = set(_dates(secondary, AAA)[50:55])  # interior hole in one feed
    secondary = secondary[~((secondary["instrument_id"] == AAA)
                            & (secondary["obs_date"].isin(gap)))]

    report = cross_vendor_report(primary, secondary)
    assert report["summary"]["instruments_flagged"] == 0
    assert report["instruments"][AAA]["flagged_dates"] == []


# ----------------------------------------------- (4) thin overlap -> set aside, unscored
def test_below_min_overlap_lands_in_insufficient_overlap():
    primary, secondary = _feeds()
    keep = set(_dates(primary, BBB)[:30])  # 30 dates -> 29 returns < min_overlap(60)

    def trunc(df):
        return df[~((df["instrument_id"] == BBB) & (~df["obs_date"].isin(keep)))]

    report = cross_vendor_report(trunc(primary), trunc(secondary))
    assert BBB in report["insufficient_overlap"]
    assert report["insufficient_overlap"][BBB] == 29
    assert BBB not in report["instruments"]
    assert AAA in report["instruments"]  # the full-history name is still scored


# --------------------------------------------------------------- (5) audit written
def test_audit_written_to_lake(tmp_lake):
    report = cross_vendor_report(*_feeds())
    path = write_cross_check_audit(report, tmp_lake)
    assert path.exists()
    assert path == tmp_lake.root / "audit" / "cross_check_prices.json"
    on_disk = json.loads(path.read_text())
    assert on_disk["summary"]["instruments_checked"] == 2


# ------------------------------------------------------------------ (6) CLI smoke test
def test_cli_cross_check_runs_on_seeded_lake(tmp_path, capsys):
    lake = Lake(tmp_path / "data")
    primary, secondary = _feeds()
    # Distinct availability so both vendors' rows coexist (curated dedup keys on
    # obs_date/instrument_id/available_from, not source).
    secondary = secondary.copy()
    secondary["available_from"] = secondary["available_from"] + pd.Timedelta(minutes=30)
    secondary["ingested_at"] = secondary["ingested_at"] + pd.Timedelta(minutes=30)
    lake.write_curated(primary, "prices", "equity")
    lake.write_curated(secondary, "prices", "equity")

    cur = lake.read_curated("prices")
    assert {"yfinance", "stooq"} <= set(cur["source"].unique())  # both feeds seeded

    rc = ingest.main(["--cross-check", "--lake-root", str(tmp_path / "data")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[cross-check]" in out
    assert "checked=" in out
    assert (lake.root / "audit" / "cross_check_prices.json").exists()


def test_cli_cross_check_graceful_when_one_vendor_absent(tmp_path, capsys):
    lake = Lake(tmp_path / "data")
    primary, _ = _feeds()  # seed only the primary feed
    lake.write_curated(primary, "prices", "equity")

    rc = ingest.main(["--cross-check", "--lake-root", str(tmp_path / "data")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "vendor" in err.lower()
