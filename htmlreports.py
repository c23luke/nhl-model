"""Backup source for shift data: the NHL's printable time-on-ice reports.

The shift-chart API (api.nhle.com/stats/rest/en/shiftcharts) sometimes returns nothing for a game,
and the same game can come back empty one hour and full the next. The printable reports
(nhl.com/scores/htmlreports/<season>/TV<game>.HTM for the visitors, TH for home) list every shift
for every player and are posted for every game. This module reads them and returns the same shape
the API gives, so parse.py does not need to know where the shifts came from.
"""
import html
import re
import unicodedata
import urllib.request

UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/126 Safari/537.36"}
URL = "https://www.nhl.com/scores/htmlreports/{season}/T{side}{num}.HTM"

HEAD = re.compile(r"^\s*(\d{1,2})\s+([^,\n]+?),\s*([^\n]+?)\s*$")
SHIFT = re.compile(r"^\s*(\d+)\s+(1|2|3|4|OT)\s+(\d{1,2}:\d\d)\s*/\s*\d{1,2}:\d\d\s+"
                   r"(\d{1,2}:\d\d)\s*/\s*\d{1,2}:\d\d\s+\d{1,2}:\d\d")


def _norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z]", "", s)


def to_text(page):
    """HTML report -> one line per table row, cells separated by spaces."""
    t = re.sub(r"(?is)<(script|style).*?</\1>", " ", page)
    t = re.sub(r"\s+", " ", t)                    # the page breaks lines inside rows; rows end at </tr>
    t = re.sub(r"(?i)</tr\s*>", "\n", t)
    t = re.sub(r"(?i)<br\s*/?>", " ", t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = html.unescape(t).replace("\xa0", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines())


def parse_text(text):
    """-> list of (sweater, last_name, period, start 'M:SS', end 'M:SS') in report order."""
    out, cur = [], None
    for ln in text.splitlines():
        m = SHIFT.match(ln)
        if m and cur:
            per = 4 if m.group(2) == "OT" else int(m.group(2))
            out.append((cur[0], cur[1], per, m.group(3), m.group(4)))
            continue
        h = HEAD.match(ln)
        if h and not ln.lower().startswith(("attendance", "start", "per ", "tot")):
            cur = (int(h.group(1)), h.group(2).strip())
    return out


def _mmss(t):
    m, s = t.split(":")
    return f"{int(m):02d}:{int(s):02d}"


def to_shiftcharts(rows_by_team, roster_spots):
    """rows_by_team: {team_id: parse_text(...)}; roster_spots: pbp['rosterSpots'].
    -> ({"data": [...], "total": n, "source": ...}, unmatched list)"""
    by_num, by_name = {}, {}
    for r in roster_spots or []:
        tid, pid = r.get("teamId"), r.get("playerId")
        if r.get("sweaterNumber") is not None:
            by_num[(tid, int(r["sweaterNumber"]))] = pid
        by_name.setdefault((tid, _norm((r.get("lastName") or {}).get("default"))), []).append(pid)
    data, unmatched = [], []
    for tid, rows in rows_by_team.items():
        for num, last, per, st, en in rows:
            pid = by_num.get((tid, num))
            if pid is None:
                c = by_name.get((tid, _norm(last)), [])
                pid = c[0] if len(c) == 1 else None
            if pid is None:
                unmatched.append((tid, num, last))
                continue
            data.append({"playerId": pid, "teamId": tid, "period": per, "startTime": _mmss(st),
                         "endTime": _mmss(en), "typeCode": 517, "eventDescription": None})
    return {"data": data, "total": len(data), "source": "nhl_html_toi_report"}, sorted(set(unmatched))


STATS = {"used": 0, "failed": 0, "why": {}}      # printed by nhl_fetch after each season


def _fail(why):
    STATS["failed"] += 1
    STATS["why"][why] = STATS["why"].get(why, 0) + 1
    return None


def fetch(season, game_id, pbp, timeout=30):
    """Shift data for one finished game from the printable reports, or None if they are not
    posted or look incomplete (fewer than 15 players with shifts on either team)."""
    num = str(game_id)[4:]
    teams = {"V": pbp["awayTeam"]["id"], "H": pbp["homeTeam"]["id"]}
    rows = {}
    for side, tid in teams.items():
        try:
            req = urllib.request.Request(URL.format(season=season, side=side, num=num), headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                page = r.read().decode("utf-8", "ignore")
        except Exception as e:
            return _fail(f"download: {type(e).__name__} {getattr(e, 'code', '')}".strip())
        rows[tid] = parse_text(to_text(page))
        if not rows[tid]:
            return _fail("report had no shift rows")
    out, unmatched = to_shiftcharts(rows, pbp.get("rosterSpots"))
    for tid in teams.values():
        if len({d["playerId"] for d in out["data"] if d["teamId"] == tid}) < 15:
            return _fail(f"under 15 players matched ({len(unmatched)} unmatched)")
    STATS["used"] += 1
    return out
