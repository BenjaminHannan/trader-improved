"""Vendor connectivity diagnostics — run this on your machine and paste
diagnostics/vendor_report.txt when ingest fails.

Stage-1 ingest depends on ~a dozen third-party vendors, several of which are
geo-fenced, rate-limited, key-gated, or prone to silently changing their payload
shape. When a loader fails on a live network we usually cannot tell *why* from the
sandbox, because the network here is blocked. This script makes ONE minimal real
request per vendor from wherever YOU run it, records exactly what came back (or the
exception), and writes a single report you can paste back.

It is deliberately dependency-light (the repo's own deps + stdlib), never raises, and
always exits 0 — a failing vendor is data, not a crash. For every vendor it prints a
one-line summary to stdout and appends a detailed block (payload type / shape /
columns / first two rows, or the exception repr) to diagnostics/vendor_report.txt.

    python scripts/diagnose_vendors.py
    # ... then paste diagnostics/vendor_report.txt
"""
from __future__ import annotations

import argparse
import datetime as _dt
import io
import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = REPO_ROOT / "diagnostics" / "vendor_report.txt"
MAX_CHARS = 2048  # truncate any raw/rendered text to ~2KB per field


# ------------------------------------------------------------------- describe
def _truncate(text: str, limit: int = MAX_CHARS) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + f"... [+{len(text) - limit} chars]"


def _is_empty(payload) -> bool:
    """True when a vendor was reachable but handed back nothing usable.

    yfinance in particular swallows a failed/geo-blocked download into an *empty*
    DataFrame (columns but zero rows) instead of raising — the exact reason a live
    ingest can 'succeed' yet parse to nothing. Flag it explicitly so the report
    distinguishes a reachable-but-empty vendor from a healthy one."""
    try:
        import pandas as pd
        if isinstance(payload, pd.DataFrame):
            return payload.empty or len(payload.columns) == 0
    except Exception:
        pass
    if isinstance(payload, (list, dict, str)):
        return len(payload) == 0
    return payload is None


def _describe(payload) -> str:
    """Render a compact, size-bounded description of a successful payload."""
    lines: list[str] = [f"type={type(payload).__name__}"]
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is not None and isinstance(payload, pd.DataFrame):
        lines.append(f"shape={payload.shape}")
        lines.append("columns=" + _truncate(repr(list(payload.columns))))
        if isinstance(payload.columns, pd.MultiIndex):
            lines.append(f"column_levels={payload.columns.nlevels} "
                         f"names={list(payload.columns.names)}")
        lines.append("index=" + _truncate(repr(list(payload.index[:3]))))
        lines.append("first_2_rows=" + _truncate(payload.head(2).to_string()))
    elif isinstance(payload, list):
        lines.append(f"len={len(payload)}")
        lines.append("first_2=" + _truncate(repr(payload[:2])))
    elif isinstance(payload, dict):
        lines.append("keys=" + _truncate(repr(list(payload.keys())[:20])))
        items = list(payload.items())[:2]
        lines.append("first_2_items=" + _truncate(repr(items)))
    else:
        lines.append("value=" + _truncate(repr(payload)))
    return "\n".join(lines)


# --------------------------------------------------------------------- probes
# Each probe returns a payload (DataFrame / list / dict / str) or raises. Kept tiny:
# one request, smallest workable window/limit, so a run is quick and cheap.
def _probe_yfinance(timeout: int):
    import yfinance as yf
    return yf.download("AAPL", period="5d", auto_adjust=True, progress=False)


def _make_ccxt_probe(exchange: str):
    def probe(timeout: int):
        import ccxt
        ex = getattr(ccxt, exchange)({"timeout": timeout * 1000})
        return ex.fetch_funding_rate_history("BTC/USDT:USDT", limit=3)
    probe.__name__ = f"ccxt_funding[{exchange}]"
    return probe


def _probe_frankfurter(timeout: int):
    import requests
    r = requests.get("https://api.frankfurter.app/2024-01-02..2024-01-05",
                     params={"from": "USD", "to": "EUR"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _probe_fred(timeout: int):
    import requests
    key = os.environ.get("FRED_API_KEY")
    if key:
        r = requests.get("https://api.stlouisfed.org/fred/series/observations",
                         params={"series_id": "DGS3MO", "api_key": key,
                                 "file_type": "json", "limit": 2,
                                 "realtime_start": "2020-01-01"}, timeout=timeout)
        r.raise_for_status()
        return {"mode": "alfred(api_key)", "sample": r.json().get("observations", [])[:2]}
    # No key: the keyless fredgraph CSV (no vintages) is what the loader falls back to.
    r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                     params={"id": "DGS3MO"}, timeout=timeout)
    r.raise_for_status()
    return {"mode": "fredgraph(no FRED_API_KEY set — no vintages)",
            "head": r.text[:400]}


def _probe_french(timeout: int):
    import requests
    url = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
           "F-F_Research_Data_Factors_daily_CSV.zip")
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        name = zf.namelist()[0]
        head = zf.read(name).decode("latin-1")[:400]
    return {"zip_member": name, "csv_head": head}


def _probe_cftc(timeout: int):
    import requests
    r = requests.get("https://publicreporting.cftc.gov/resource/6dca-aqww.json",
                     params={"$limit": 2}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _probe_kalshi(timeout: int):
    import requests
    r = requests.get("https://api.elections.kalshi.com/trade-api/v2/markets",
                     params={"limit": 2, "status": "active"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _probe_polymarket(timeout: int):
    import requests
    r = requests.get("https://gamma-api.polymarket.com/markets",
                     params={"limit": 2, "closed": "false", "order": "volume",
                             "ascending": "false"}, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _probe_coingecko(timeout: int):
    import requests
    headers = {}
    key = os.environ.get("COINGECKO_API_KEY")
    if key:
        headers["x-cg-demo-api-key"] = key
    r = requests.get("https://api.coingecko.com/api/v3/coins/markets",
                     params={"vs_currency": "usd", "order": "market_cap_desc",
                             "per_page": 2, "page": 1}, headers=headers or None,
                     timeout=timeout)
    r.raise_for_status()
    return r.json()


def _probe_defillama(timeout: int):
    import requests
    r = requests.get("https://api.llama.fi/v2/chains", timeout=timeout)
    r.raise_for_status()
    return r.json()[:2]


def _probe_edgar(timeout: int):
    import requests
    ua = os.environ.get("SEC_USER_AGENT",
                        "trader-improved research benjamin.a.hannan@gmail.com")
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers={"User-Agent": ua, "Accept-Encoding": "gzip, deflate"},
                     timeout=timeout)
    r.raise_for_status()
    data = r.json()
    sample = dict(list(data.items())[:2]) if isinstance(data, dict) else data[:2]
    return {"count": len(data), "sample": sample}


def _build_probes():
    from production.data.loaders.ccxt_funding import DEFAULT_EXCHANGE_CHAIN

    probes: list[tuple[str, callable]] = [("yfinance", _probe_yfinance)]
    for exch in DEFAULT_EXCHANGE_CHAIN:
        probes.append((f"ccxt_funding[{exch}]", _make_ccxt_probe(exch)))
    probes += [
        ("frankfurter", _probe_frankfurter),
        ("fred", _probe_fred),
        ("french", _probe_french),
        ("cftc", _probe_cftc),
        ("kalshi", _probe_kalshi),
        ("polymarket", _probe_polymarket),
        ("coingecko", _probe_coingecko),
        ("defillama", _probe_defillama),
        ("edgar", _probe_edgar),
    ]
    return probes


# ----------------------------------------------------------------------- main
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Probe every Stage-1 vendor with one minimal request and write "
                    "diagnostics/vendor_report.txt (paste it when ingest fails).")
    p.add_argument("--output", default=str(DEFAULT_REPORT),
                   help="report file to write (default: diagnostics/vendor_report.txt)")
    p.add_argument("--timeout", type=int, default=30, help="per-request timeout (s)")
    args = p.parse_args(argv)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat()

    blocks = [
        "=" * 72,
        "trader-improved vendor diagnostics",
        f"generated: {stamp}",
        "paste this file when a Stage-1 ingest loader fails.",
        "=" * 72,
        "",
    ]

    try:
        probes = _build_probes()
    except Exception as exc:  # even wiring must not crash the diagnostic
        probes = []
        blocks.append(f"[setup] FAILED to build probe list: {exc!r}\n")

    for name, probe in probes:
        block = [f"### {name}"]
        try:
            payload = probe(args.timeout)
            empty = _is_empty(payload)
            status = "OK (empty payload — vendor reachable but returned no rows)" \
                if empty else "OK"
            block.append("status: " + status)
            block.append(_describe(payload))
            print(f"[{name}] {status}")
        except Exception as exc:  # noqa: BLE001 — a failing vendor is expected data
            block.append("status: FAIL")
            block.append("exception: " + _truncate(repr(exc)))
            print(f"[{name}] FAIL: {exc!r}")
        block.append("")
        blocks.append("\n".join(block))

    try:
        out_path.write_text("\n".join(blocks), encoding="utf-8")
        print(f"\nwrote {out_path}")
    except Exception as exc:  # last-resort: report path unwritable
        print(f"\nfailed to write {out_path}: {exc!r}")

    return 0  # always succeed — this is a diagnostic, not a gate


if __name__ == "__main__":
    raise SystemExit(main())
