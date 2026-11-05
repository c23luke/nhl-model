"""Tonight's lineups and starting goalies.

DailyFaceoff team pages carry the projected lines as structured data (forward lines f1-f4,
pairs d1-d3, goalies g1/g2, injuries, last-updated time and source). NHL rosters give player ids.
The starting-goalie page says whether each starter is Confirmed.
"""
import json
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126 Safari/537.36"}

DFO_SLUG = {
    "ANA": "anaheim-ducks", "BOS": "boston-bruins", "BUF": "buffalo-sabres", "CGY": "calgary-flames",
    "CAR": "carolina-hurricanes", "CHI": "chicago-blackhawks", "COL": "colorado-avalanche",
    "CBJ": "columbus-blue-jackets", "DAL": "dallas-stars", "DET": "detroit-red-wings",
    "EDM": "edmonton-oilers", "FLA": "florida-panthers", "LAK": "los-angeles-kings", "MIN": "minnesota-wild",
    "MTL": "montreal-canadiens", "NSH": "nashville-predators", "NJD": "new-jersey-devils",
    "NYI": "new-york-islanders", "NYR": "new-york-rangers", "OTT": "ottawa-senators",
    "PHI": "philadelphia-flyers", "PIT": "pittsburgh-penguins", "SJS": "san-jose-sharks",
    "SEA": "seattle-kraken", "STL": "st-louis-blues", "TBL": "tampa-bay-lightning",
    "TOR": "toronto-maple-leafs", "UTA": "utah-mammoth", "VAN": "vancouver-canucks",
    "VGK": "vegas-golden-knights", "WSH": "washington-capitals", "WPG": "winnipeg-jets",
}


def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).strip()


def get(url, as_json=False):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read().decode("utf-8", "ignore")
    return json.loads(body) if as_json else body


def next_data(html):
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    return json.loads(m.group(1)) if m else None


# ---------------------------------------------------------------- lines
def parse_lines(html):
    """-> {'forwards': [names x12], 'defense': [names x6], 'goalies': [g1, g2], 'injured': [...],
           'updated_at': datetime, 'source': str}"""
    d = next_data(html)
    if not d:
        return None
    c = d["props"]["pageProps"].get("combinations") or {}
    out = {"forwards": [], "defense": [], "goalies": [], "injured": [], "source": c.get("sourceName"),
           "updated_at": None, "team": c.get("teamAbbreviation"), "meta": {}}
    if c.get("updatedAt"):
        out["updated_at"] = datetime.fromisoformat(c["updatedAt"].replace("Z", "+00:00"))
    order = {"f1": 0, "f2": 1, "f3": 2, "f4": 3, "d1": 0, "d2": 1, "d3": 2}
    for p in c.get("players", []):
        g, cat = p.get("groupIdentifier"), p.get("categoryIdentifier")
        if cat == "ev" and p.get("name"):
            out["meta"][p["name"]] = {"num": p.get("jerseyNumber"), "gtd": bool(p.get("gameTimeDecision")),
                                      "injury": p.get("injuryStatus")}
        if cat == "ev" and g in ("f1", "f2", "f3", "f4"):
            out["forwards"].append((order[g], p["name"]))
        elif cat == "ev" and g in ("d1", "d2", "d3"):
            out["defense"].append((order[g], p["name"]))
        elif cat == "ev" and g == "g":
            out["goalies"].append((p.get("positionIdentifier"), p["name"]))
        elif cat == "oi":
            out["injured"].append({"name": p["name"], "status": p.get("injuryStatus")})
    out["forwards"] = [n for _, n in sorted(out["forwards"])]
    out["defense"] = [n for _, n in sorted(out["defense"])]
    out["goalies"] = [n for _, n in sorted(out["goalies"])]
    return out


def fetch_lines(abbrev):
    return parse_lines(get(f"https://www.dailyfaceoff.com/teams/{DFO_SLUG[abbrev]}/line-combinations"))


# ---------------------------------------------------------------- starting goalies
def parse_starting_goalies(html):
    """-> {team_abbrev: {'name': str, 'confirmed': bool, 'source': str|None, 'at': str|None}}
    Built against DailyFaceoff's page data. If the layout is not recognised it returns {},
    which keeps every pick waiting rather than guessing."""
    d = next_data(html)
    if not d:
        return {}
    out = {}

    def walk(o):
        if isinstance(o, dict):
            keys = set(o)
            # a game entry: home/away goalie name + status fields
            for side in ("home", "away"):
                nk, sk, tk = f"{side}GoalieName", f"{side}NewsStrengthName", f"{side}TeamAbbreviation"
                if nk in keys and tk in keys:
                    status = str(o.get(sk) or "")
                    out[o[tk]] = {"name": o[nk], "confirmed": status.lower() == "confirmed",
                                  "status": status, "source": o.get(f"{side}NewsSourceName"),
                                  "at": o.get(f"{side}NewsCreatedAt")}
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(d)
    return out


def fetch_starting_goalies(date_et):
    return parse_starting_goalies(get(f"https://www.dailyfaceoff.com/starting-goalies/{date_et}"))


# ---------------------------------------------------------------- NHL ids
def roster_ids(abbrev):
    """{normalised name: (player_id, pos)} from the NHL's current roster."""
    r = get(f"https://api-web.nhle.com/v1/roster/{abbrev}/current", as_json=True)
    out = {}
    for grp, pos in (("forwards", "F"), ("defensemen", "D"), ("goalies", "G")):
        for p in r.get(grp, []):
            name = f"{p['firstName']['default']} {p['lastName']['default']}"
            out[norm(name)] = (int(p["id"]), pos)
    return out


def match(name, ids):
    """Exact normalised match, then last name + first initial. None if not found."""
    n = norm(name)
    if n in ids:
        return ids[n]
    parts = n.split()
    if len(parts) >= 2:
        cands = [v for k, v in ids.items() if k.split()[-1] == parts[-1] and k[:1] == parts[0][:1]]
        if len(cands) == 1:
            return cands[0]
    return None


POS = {"C": "F", "L": "F", "R": "F", "D": "D", "G": "G"}


def spot_ids(spots):
    """NHL pregame rosterSpots for one team -> ({jersey: (id, pos)}, {normalised name: (id, pos)}).
    This is tonight's active roster (dressed players plus healthy scratches), so it includes
    call-ups and new arrivals that the 'current roster' list can lag on."""
    by_num, by_name = {}, {}
    for r in spots or []:
        v = (int(r["playerId"]), POS.get(r.get("positionCode"), "F"))
        if r.get("sweaterNumber") is not None:
            by_num[int(r["sweaterNumber"])] = v
        by_name[norm(f"{r['firstName']['default']} {r['lastName']['default']}")] = v
    return by_num, by_name


def team_lineup(abbrev, puck_drop_utc, spots=None):
    """Everything the board needs for one team, with the evidence for each check.
    spots: tonight's NHL pregame rosterSpots for this team (optional)."""
    lines = fetch_lines(abbrev)
    if not lines:
        return {"ok": False, "why": f"{abbrev} lines page unreadable"}
    try:
        ids = roster_ids(abbrev)
    except Exception:
        ids = {}
    by_num, by_name = spot_ids(spots)
    for k, v in by_name.items():
        ids.setdefault(k, v)
    meta = lines.get("meta", {})

    def find(name):
        # jersey number on tonight's roster first (no spelling problems), then the name
        num = (meta.get(name) or {}).get("num")
        if num is not None and int(num) in by_num:
            return by_num[int(num)]
        return match(name, ids)

    skaters, missing = [], []
    for name in lines["forwards"]:
        m = find(name)
        (skaters.append((m[0], "F")) if m else missing.append(name))
    for name in lines["defense"]:
        m = find(name)
        (skaters.append((m[0], "D")) if m else missing.append(name))
    g1 = find(lines["goalies"][0]) if lines["goalies"] else None
    hours_old = None
    if lines["updated_at"]:
        hours_old = (puck_drop_utc - lines["updated_at"]).total_seconds() / 3600
    fresh = hours_old is not None and hours_old <= 14          # updated since this morning's skate
    flagged = [n for n in lines["forwards"] + lines["defense"]
               if (meta.get(n) or {}).get("gtd") or (meta.get(n) or {}).get("injury")]
    tonight = {v[0] for v in by_num.values()} | {v[0] for v in by_name.values()}
    on_roster = bool(spots) and all(pid in tonight for pid, _ in skaters)
    return {"ok": True, "skaters": skaters, "missing": missing, "projected_goalie": lines["goalies"][:1],
            "projected_goalie_id": g1[0] if g1 else None, "updated_at": lines["updated_at"],
            "hours_old": hours_old, "fresh": fresh, "source": lines["source"],
            "complete": not missing and len(skaters) == 18 and len(lines["defense"]) >= 6,
            "n_listed": (len(lines["forwards"]), len(lines["defense"])),
            "flagged": flagged, "on_tonight_roster": on_roster,
            "ids": ids}


def stale_ok(lu, goalie_confirmed, max_hours=96):
    """A lines page that hasn't been touched today is still trusted when nothing suggests a change:
    updated within the last 4 days, the team's starting goalie confirmed today, no player in the
    18 marked injured or a game-time decision, and all 18 on the NHL's roster for tonight's game."""
    h = lu.get("hours_old")
    return (not lu["fresh"] and h is not None and h <= max_hours and goalie_confirmed
            and not lu["flagged"] and lu["on_tonight_roster"])
