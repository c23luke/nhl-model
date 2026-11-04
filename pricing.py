"""Line shopping across Caden's books, after each book's fee.

Takes an Odds API response (live or historical) and, for every game and side, returns:
  - every book's price, as posted and after that book's fee
  - the best book to bet (highest payout after fee)
  - the market's fair win chance (each book's no-vig price, then the median)

Fees are pregame taker fees as published in Nov 2026. Edit FEES if a book changes them.
"""
from statistics import median

BOOKS = ["draftkings", "bovada", "novig", "prophetx", "kalshi", "polymarket", "polymarket_us", "betopenly"]
BOOK_NAMES = {"draftkings": "DraftKings", "bovada": "Bovada", "novig": "Novig", "prophetx": "ProphetX",
              "kalshi": "Kalshi", "polymarket": "Polymarket", "polymarket_us": "Polymarket US",
              "betopenly": "BetOpenly"}

# kind: "none"      price already includes the book's cut
#       "winnings"  rate taken from net winnings
#       "stake"     rate taken from the stake (used where it is unclear which, as the costlier reading)
#       "contract"  prediction-market taker fee per $1 contract = rate * p * (1 - p)
FEES = {
    "draftkings":    ("none", 0.0),
    "bovada":        ("none", 0.0),
    "novig":         ("none", 0.0),       # no fee on pregame straight bets
    "prophetx":      ("winnings", 0.02),
    "kalshi":        ("contract", 0.07),
    "polymarket":    ("contract", 0.03),
    "polymarket_us": ("contract", 0.05),
    "betopenly":     ("stake", 0.01),     # 1% fee; unconfirmed whether on stake or winnings
}


def to_decimal(american):
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def to_american(dec):
    return int(round((dec - 1) * 100)) if dec >= 2 else int(round(-100 / (dec - 1)))


def after_fee(book, dec):
    """Decimal payout per $1 staked after the book's fee."""
    kind, rate = FEES.get(book, ("none", 0.0))
    if kind == "winnings":
        return 1 + (dec - 1) * (1 - rate)
    if kind == "stake":
        return dec / (1 + rate)
    if kind == "contract":
        p = 1 / dec                                  # contract price per $1 payout
        return 1 / (p + rate * p * (1 - p))
    return dec


def _side_prices(event, market_key, side_name, point=None, who=None):
    out = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != market_key:
                continue
            for o in mk.get("outcomes", []):
                if o["name"] != side_name:
                    continue
                if point is not None and o.get("point") != point:
                    continue
                if who is not None and o.get("description") != who:
                    continue
                dec = to_decimal(o["price"])
                eff = after_fee(bk["key"], dec)
                out.append({"book": bk["key"], "name": BOOK_NAMES.get(bk["key"], bk["key"]),
                            "american": int(o["price"]), "decimal": round(dec, 4),
                            "after_fee": round(eff, 4), "after_fee_american": to_american(eff),
                            "limit": o.get("bet_limit"), "updated": mk.get("last_update")})
    return sorted(out, key=lambda r: -r["after_fee"])


def _fair(event, market_key, a, b, point=None, who=None):
    """Median of each book's no-vig chance for side a, using posted prices."""
    probs = []
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != market_key:
                continue
            o = {}
            for x in mk.get("outcomes", []):
                if (point is None or x.get("point") == point) and (who is None or x.get("description") == who):
                    o[x["name"]] = x
            if a in o and b in o:
                pa, pb = 1 / to_decimal(o[a]["price"]), 1 / to_decimal(o[b]["price"])
                probs.append(pa / (pa + pb))
    return (median(probs), len(probs)) if probs else (None, 0)


def shop_moneyline(event):
    h, a = event["home_team"], event["away_team"]
    fair, n = _fair(event, "h2h", h, a)
    return {"market": "moneyline", "books_priced": n, "fair_home": fair,
            "home": _side_prices(event, "h2h", h), "away": _side_prices(event, "h2h", a)}


def shop_total(event):
    pts = [o.get("point") for bk in event.get("bookmakers", []) for mk in bk.get("markets", [])
           if mk["key"] == "totals" for o in mk.get("outcomes", []) if o["name"] == "Over"]
    if not pts:
        return None
    line = max(set(pts), key=pts.count)               # the line most books are on
    fair, n = _fair(event, "totals", "Over", "Under", point=line)
    return {"market": "total", "line": line, "books_priced": n, "fair_over": fair,
            "over": _side_prices(event, "totals", "Over", point=line),
            "under": _side_prices(event, "totals", "Under", point=line)}


def shop_props(event, market_key="player_shots_on_goal"):
    """{player: {line, fair_over, over:[...], under:[...]}} using each player's most common line."""
    seen = {}
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            if mk["key"] != market_key:
                continue
            for o in mk.get("outcomes", []):
                if o.get("description"):
                    seen.setdefault(o["description"], []).append(o.get("point"))
    out = {}
    for who, pts in seen.items():
        line = max(set(pts), key=pts.count)
        fair, n = _fair(event, market_key, "Over", "Under", point=line, who=who)
        out[who] = {"line": line, "books_priced": n, "fair_over": fair,
                    "over": _side_prices(event, market_key, "Over", point=line, who=who),
                    "under": _side_prices(event, market_key, "Under", point=line, who=who)}
    return out


def ladder_text(rows, top=4):
    """Short plain-words price ladder for the dashboard."""
    parts = []
    for r in rows[:top]:
        fee = "" if r["american"] == r["after_fee_american"] else f" ({r['after_fee_american']:+d} after fee)"
        parts.append(f"{r['name']} {r['american']:+d}{fee}")
    return ", ".join(parts)


# ---------------------------------------------------------------- how to place it
UNIT_TO_WIN = 20.0          # one unit = to win $20 (Caden, Nov 2026)


def fill_plan(rows, units, unit_to_win=UNIT_TO_WIN):
    """Split one bet across books, best payout after fee first.
    Fills as much as each book allows (its posted max stake) before moving to the next.
    rows: a side's price list from shop_* (already sorted best first).
    Returns [{book, name, american, after_fee_american, risk, to_win}] and the amount left unfilled."""
    target = units * unit_to_win                 # net winnings wanted, after fees
    plan, left = [], target
    for r in rows:
        if left <= 0.005:
            break
        eff = r["after_fee"]
        risk_needed = left / (eff - 1)
        cap = r.get("limit")
        risk = min(risk_needed, float(cap)) if cap not in (None, "", 0) else risk_needed
        if risk < 1:                               # books don't take bets under $1
            continue
        won = risk * (eff - 1)
        plan.append({"book": r["book"], "name": r["name"], "american": r["american"],
                     "after_fee_american": r["after_fee_american"],
                     "risk": round(risk, 2), "to_win": round(won, 2)})
        left -= won
    return plan, round(max(left, 0), 2)
