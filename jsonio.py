"""Read/write the cached API files gzip-compressed (~10x smaller than plain json)."""
import gzip
import json
from pathlib import Path


def gz(path):
    p = Path(path)
    return p if p.name.endswith(".gz") else p.with_name(p.name + ".gz")


def save(path, data):
    p = gz(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(p)                      # never leaves a half-written file behind


def load(path):
    p = gz(path)
    if p.exists():
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    plain = Path(str(p)[:-3])
    return json.loads(plain.read_text())


def exists(path):
    p = gz(path)
    return p.exists() or Path(str(p)[:-3]).exists()


def compact(folder):
    """Convert every plain .json under folder to .json.gz and delete the original.
    A file that will not parse (cut off when the disk filled) is deleted so it re-downloads."""
    files = list(Path(folder).rglob("*.json"))
    bad = 0
    for i, f in enumerate(files):
        try:
            data = json.loads(f.read_text())
        except Exception:
            f.unlink()
            bad += 1
            continue
        save(f, data)
        f.unlink()
        if i % 500 == 0:
            print(f"  compressing cached files {i}/{len(files)}", end="\r", flush=True)
    if files:
        print(f"  compressed {len(files) - bad} files, removed {bad} broken ones          ")
