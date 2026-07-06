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
