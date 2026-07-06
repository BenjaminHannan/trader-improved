"""Curated-zone schema enforcement and vintage semantics for `production.core.lake`.

These tests pin the lake's non-negotiable guarantees: nothing lands in the curated
zone without the mandatory point-in-time columns, an entity key, and non-null
availability/observation stamps; and re-runs are idempotent with latest-vintage-wins
upsert per key. Year partitioning on disk is verified because incremental reads and
`watermark` rely on it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from production.core.lake import LakeError
from tests.conftest import make_gbm_prices, make_macro

INSTR = {"EQ:SYN00:2000-01-03": "equity"}


def _prices(start="2020-01-02", end="2020-01-10") -> pd.DataFrame:
    """Small valid curated-format price frame via the shared maker."""
    return make_gbm_prices(INSTR, start=start, end=end)


# ---------------------------------------------------------------- schema rejects
@pytest.mark.parametrize("missing", ["obs_date", "available_from", "source", "ingested_at"])
def test_write_curated_rejects_missing_mandatory_column(tmp_lake, missing):
    df = _prices().drop(columns=[missing])
    with pytest.raises(LakeError):
        tmp_lake.write_curated(df, "prices", "equity")


def test_write_curated_rejects_missing_entity_column(tmp_lake):
    # Neither instrument_id nor series_id present -> no entity key.
    df = _prices().drop(columns=["instrument_id"])
    with pytest.raises(LakeError):
        tmp_lake.write_curated(df, "prices", "equity")


def test_write_curated_rejects_null_available_from(tmp_lake):
    df = _prices()
    df.loc[df.index[0], "available_from"] = pd.NaT
    with pytest.raises(LakeError):
        tmp_lake.write_curated(df, "prices", "equity")


def test_write_curated_rejects_null_obs_date(tmp_lake):
    df = _prices()
    df.loc[df.index[0], "obs_date"] = pd.NaT
    with pytest.raises(LakeError):
        tmp_lake.write_curated(df, "prices", "equity")


# ---------------------------------------------------------- accept + roundtrip
def test_write_curated_accepts_valid_frame_and_roundtrips(tmp_lake):
    df = _prices()
    tmp_lake.write_curated(df, "prices", "equity")
    back = tmp_lake.read_curated("prices", "equity")
    assert len(back) == len(df)
    # Same rows returned (compare on the identifying + value columns, order-agnostic).
    lhs = df.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)
    rhs = back.sort_values(["obs_date", "instrument_id"]).reset_index(drop=True)
    assert (lhs["instrument_id"].tolist() == rhs["instrument_id"].tolist())
    assert lhs["close"].round(9).tolist() == rhs["close"].round(9).tolist()
    assert lhs["obs_date"].tolist() == rhs["obs_date"].tolist()


def test_write_curated_partitions_by_year_on_disk(tmp_lake):
    # Frame spans a year boundary -> two year=YYYY partition directories.
    df = _prices(start="2019-12-27", end="2020-01-08")
    tmp_lake.write_curated(df, "prices", "equity")
    base = tmp_lake.root / "curated" / "dataset=prices" / "asset_class=equity"
    year_dirs = sorted(p.name for p in base.glob("year=*"))
    assert year_dirs == ["year=2019", "year=2020"]
    for yd in year_dirs:
        assert (base / yd / "part.parquet").exists()


# ------------------------------------------------------------ idempotence/upsert
def test_write_curated_is_idempotent(tmp_lake):
    df = _prices()
    tmp_lake.write_curated(df, "prices", "equity")
    tmp_lake.write_curated(df, "prices", "equity")  # identical re-run
    back = tmp_lake.read_curated("prices", "equity")
    assert len(back) == len(df)  # no duplication of the same vintage


def test_write_curated_vintage_upsert_keeps_later_ingest(tmp_lake):
    df = _prices()
    tmp_lake.write_curated(df, "prices", "equity")

    # Same (obs_date, instrument_id, available_from) key, revised value, later ingest.
    key = df.iloc[[0]].copy()
    key["close"] = 999.0
    key["ingested_at"] = key["ingested_at"] + pd.Timedelta(hours=1)
    tmp_lake.write_curated(key, "prices", "equity")

    back = tmp_lake.read_curated("prices", "equity")
    assert len(back) == len(df)  # upsert, not append
    row = back[(back["obs_date"] == df.iloc[0]["obs_date"])
               & (back["instrument_id"] == df.iloc[0]["instrument_id"])]
    assert len(row) == 1
    assert row["close"].iloc[0] == 999.0  # later ingested_at wins


# ------------------------------------------------------------------- watermark
def test_watermark_returns_max_obs_date(tmp_lake):
    df = _prices()
    tmp_lake.write_curated(df, "prices", "equity")
    wm = tmp_lake.watermark("prices", "equity")
    assert wm == pd.to_datetime(df["obs_date"]).max()


def test_watermark_none_for_unknown_dataset(tmp_lake):
    assert tmp_lake.watermark("does_not_exist") is None


# ----------------------------------------------------------- start/end filtering
def test_read_curated_start_end_filtering(tmp_lake):
    df = _prices(start="2019-06-03", end="2020-06-30")
    tmp_lake.write_curated(df, "prices", "equity")

    lo, hi = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-03-31")
    win = tmp_lake.read_curated("prices", "equity", start=lo, end=hi)
    assert not win.empty
    assert win["obs_date"].min() >= lo
    assert win["obs_date"].max() <= hi
    # And it is a strict subset of the full dataset.
    full = tmp_lake.read_curated("prices", "equity")
    assert len(win) < len(full)


def test_read_curated_series_id_entity_roundtrips(tmp_lake):
    """Macro frames key on series_id (the other allowed entity column)."""
    df = make_macro({"DGS3MO_US": 2.0}, start="2020-01-02", end="2020-01-10")
    tmp_lake.write_curated(df, "macro", "macro")
    back = tmp_lake.read_curated("macro", "macro")
    assert len(back) == len(df)
    assert set(back["series_id"].unique()) == {"DGS3MO_US"}
