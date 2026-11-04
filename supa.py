"""Tiny Supabase REST client (secret key from the SUPABASE_SERVICE_KEY environment variable)."""
import json
import os
import urllib.parse
import urllib.request

URL = os.environ.get("SUPABASE_URL", "https://pbnydggxfqsmstosneth.supabase.co") + "/rest/v1"


def _headers(extra=None):
    key = os.environ["SUPABASE_SERVICE_KEY"]
    h = {"apikey": key, "Content-Type": "application/json"}
    if key.startswith("eyJ"):
        h["Authorization"] = f"Bearer {key}"
    h.update(extra or {})
    return h


def select(table, params):
    q = urllib.parse.urlencode(params, safe="(),.*:")
    req = urllib.request.Request(f"{URL}/{table}?{q}", headers=_headers())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def upsert(table, rows, on_conflict):
    for i in range(0, len(rows), 300):
        req = urllib.request.Request(
            f"{URL}/{table}?on_conflict={on_conflict}", data=json.dumps(rows[i:i + 300], default=str).encode(),
            headers=_headers({"Prefer": "resolution=merge-duplicates,return=minimal"}), method="POST")
        urllib.request.urlopen(req, timeout=60)


def patch(table, filters, values):
    q = urllib.parse.urlencode(filters, safe="(),.*:")
    req = urllib.request.Request(f"{URL}/{table}?{q}", data=json.dumps(values, default=str).encode(),
                                 headers=_headers({"Prefer": "return=minimal"}), method="PATCH")
    urllib.request.urlopen(req, timeout=60)
