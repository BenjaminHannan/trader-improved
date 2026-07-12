"""Per-sleeve trading calendars + the shared weekly rebalance grid.

NYSE calendar is approximated with US federal holidays (pandas has no bundled NYSE
calendar; Good Friday is added by hand since it is an NYSE holiday but not federal).
Good enough for a daily-bar, weekly-rebalance system — documented approximation.
Crypto trades 7 days a week on midnight-UTC bars.
"""
from __future__ import annotations

import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
)
from pandas.tseries.offsets import CustomBusinessDay


class NYSEApproxCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-06-19",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


_NYSE_BDAY = CustomBusinessDay(calendar=NYSEApproxCalendar())


def trading_days(sleeve_calendar: str, start, end) -> pd.DatetimeIndex:
    """All trading days for a sleeve calendar in [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if sleeve_calendar == "crypto":
        return pd.date_range(start, end, freq="D")
    if sleeve_calendar == "nyse":
        return pd.date_range(start, end, freq=_NYSE_BDAY)
    raise ValueError(f"unknown calendar {sleeve_calendar!r}")


def rebalance_grid(start, end, freq: str = "weekly") -> pd.DatetimeIndex:
    """Shared rebalance grid across all sleeves.

    Weekly = the last NYSE trading day of each ISO week (normally Friday). Crypto
    trades on the same grid — daily bars, weekly trades, v1 simplicity.
    """
    days = trading_days("nyse", start, end)
    if freq == "weekly":
        s = pd.Series(days, index=days)
        return pd.DatetimeIndex(s.groupby([days.isocalendar().year, days.isocalendar().week],
                                          sort=False).last().values)
    if freq == "daily":
        return days
    if freq == "monthly":
        s = pd.Series(days, index=days)
        return pd.DatetimeIndex(s.groupby([days.year, days.month], sort=False).last().values)
    raise ValueError(f"unknown rebalance freq {freq!r}")


def month_starts(grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First grid date of each calendar month — the monthly re-estimation points."""
    s = pd.Series(grid, index=grid)
    return pd.DatetimeIndex(s.groupby([grid.year, grid.month], sort=False).first().values)


def twice_weekly_grid(start, end) -> pd.DatetimeIndex:
    """Twice-weekly NYSE grid: the weekly grid PLUS the trading day closest to each Monday.

    A middle rebalance step between weekly and daily (the crypto sleeve uses it — 30bp floors
    make a full daily cadence marginal at v1 alpha strength). Monday is normally a trading day;
    on a Monday holiday the nearest NYSE trading day is used (the following Tuesday, or the
    prior Friday if it is closer). Dedup-unioned with the weekly grid, so a Monday that already
    coincides with a weekly grid date collapses to one entry.
    """
    weekly = rebalance_grid(start, end, "weekly")
    days = trading_days("nyse", start, end)
    if len(days) == 0:
        return weekly
    mondays = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq="W-MON")
    extra = []
    for m in mondays:
        pos = int(days.searchsorted(m))
        cand = []
        if pos < len(days):
            cand.append(days[pos])
        if pos > 0:
            cand.append(days[pos - 1])
        if cand:
            extra.append(min(cand, key=lambda d: abs((d - m).days)))
    grid = weekly.union(pd.DatetimeIndex(extra))
    return pd.DatetimeIndex(grid.sort_values())


def offset_grid(grid: pd.DatetimeIndex, offset_days: int,
                calendar: str = "nyse") -> pd.DatetimeIndex:
    """Shift each grid date forward by ``offset_days`` TRADING days on ``calendar``.

    Used to build the tranche decision grids (offsets 0..K-1). Dates whose shifted position
    would fall beyond the calendar's available range are dropped (clipped at the end).
    ``offset_days == 0`` returns the grid unchanged (a bit-identical passthrough).
    """
    grid = pd.DatetimeIndex(grid)
    if len(grid) == 0 or offset_days == 0:
        return grid
    lo = pd.Timestamp(grid.min())
    hi = pd.Timestamp(grid.max()) + pd.Timedelta(days=offset_days * 4 + 10)
    cal = trading_days(calendar, lo, hi)
    pos = cal.searchsorted(grid)
    new_pos = pos + offset_days
    valid = new_pos < len(cal)
    return pd.DatetimeIndex(cal[new_pos[valid]])
