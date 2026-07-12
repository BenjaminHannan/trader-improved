"""Maker upper-bound re-price (registration 003ded7): the two filed negatives
at 25%-of-taker fees, assuming 100% fill and zero adverse selection.

Per strategy: net edge <= 0 => maker path PERMANENTLY CLOSED (loses even with
the free-est possible execution); > 0 => candidate CONTINGENT on fill
calibration (nothing promotable without fill data). Parameters frozen to the
original registrations (longshot P2 from iteration 5; post-move fade from
iteration 9). Verdict-only output.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from _common import cluster_mean_test, taker_fee  # noqa: E402
from kalshi_longshot_fade import FADE_BUCKET, build_entries, load_panel  # noqa: E402
from kalshi_postmove_drift import build_triggers  # noqa: E402

MAKER_FRAC = 0.25


def maker_post_fee_return(entry: float, terminal: float) -> float:
    """Whelan eq. (3) with the maker fee (25% of taker) as the commission."""
    c = MAKER_FRAC * taker_fee(entry)
    return (terminal - entry - c) / (entry + c)


def main() -> int:
    # ---------------------------------------------- (a) longshot fade P2
    bars, st = load_panel()
    ent = build_entries(bars, st)
    p2 = ent[(ent["entry_yes"] >= FADE_BUCKET[0]) & (ent["entry_yes"] <= FADE_BUCKET[1])]
    ret = p2.apply(lambda r: maker_post_fee_return(1.0 - r["entry_yes"],
                                                   1.0 - r["terminal"]), axis=1)
    s = cluster_mean_test(ret, p2["event_key"])
    a_open = s["mean"] > 0
    print(f"(a) longshot fade P2 @ maker fees: n={s['n']} mean={s['mean']:+.4f} "
          f"t={s['t']:.2f} -> {'CANDIDATE (contingent on fill data)' if a_open else 'PERMANENTLY CLOSED'}")

    # ---------------------------------------------- (b) post-move fade
    trig = build_triggers()
    fees_taker2 = trig["cont"] - trig["net"]           # the x2 taker fees charged there
    fade_net = -trig["cont"] - MAKER_FRAC * fees_taker2
    s2 = cluster_mean_test(fade_net, trig["event_key"])
    b_open = s2["mean"] > 0
    print(f"(b) post-move fade @ maker x2 fees: n={s2['n']} mean={s2['mean']:+.4f} "
          f"t={s2['t']:.2f} -> {'CANDIDATE (contingent on fill data)' if b_open else 'PERMANENTLY CLOSED'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
