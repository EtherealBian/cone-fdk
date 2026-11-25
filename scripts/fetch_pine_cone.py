"""Acquire pinned Zenodo files of the pine cone. Single writer; resume HTTP ranges, then verify whole originals."""

import argparse
import hashlib
import http.client
import json
import os
import re
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "0e236a75a8d4ebe9dc09b1f1a79f8ab832c42df8738340d66b40ba13d4c5deb4"
META = "20201118_pine_cone_.txt"
STEP = 1 << 20
RANGE = 16 << 20


def need(value, why):
    if not value:
        raise ValueError(why)


def hashes(path):
    m, h = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as f:
        while b := f.read(STEP):
            m.update(b)
            h.update(b)
    return m.hexdigest(), h.hexdigest()


def source():
    raw = (ROOT / "fixtures/pine-cone-source-v1.json").read_bytes()
    need(hashlib.sha256(raw).hexdigest() == SOURCE, "different pine source manifest")
    d = json.loads(raw)
    need(
        d["record_id"] == 6985407
        and d["license"] == "CC BY 4.0"
        and set(d["files"])
        == {META, *(f"20201118_pine_cone_projections_{i}.zip" for i in range(1, 9))},
        "unexpected record or file set",
    )
    return d


def block(url, begin, end, size, opener):
    # Range-specific cache key prevents proxies reusing a previous 206 for another interval.
    req = urllib.request.Request(
        f"{url}?bytes={begin}-{end}",
        headers={
            "User-Agent": "ConeFDK-PineCone-Verification/1",
            "Range": f"bytes={begin}-{end}",
            "Accept-Encoding": "identity",
        },
    )
    with opener(req, timeout=100) as f:
        complete = begin == 0 and end + 1 == size
        need(
            (f.status == 206 and f.headers.get("Content-Range") == f"bytes {begin}-{end}/{size}")
            or (complete and f.status == 200),
            "unmatched HTTP range; not appending",
        )
        need(
            f.headers.get("Content-Encoding", "identity") == "identity",
            "unexpected transport encoding",
        )
        n = end - begin + 1
        if length := f.headers.get("Content-Length"):
            need(length == str(n), "unexpected HTTP length")
        buf = f.read(n + 1)
        need(len(buf) == n, "short or oversized response")
        return buf


def obtain(folder, name, rec, opener=urllib.request.urlopen, chunk=RANGE):
    need(
        name == META or re.fullmatch(r"20201118_pine_cone_projections_[1-8]\.zip", name),
        "unexpected source name",
    )
    n, md = rec["bytes"], rec["md5"]
    final = folder / name
    partial = folder / (name + ".partial.download")
    if final.exists():
        need(
            final.is_file() and final.stat().st_size == n,
            "existing original has wrong length; refusing overwrite",
        )
        m, h = hashes(final)
        need(m == md, "existing original differs from publisher checksum")
        return h
    start = partial.stat().st_size if partial.exists() else 0
    need(
        0 <= start <= n and (start == n or start % chunk == 0),
        "partial ends between committed ranges; move it aside before retrying",
    )
    url = f"https://zenodo.org/api/records/6985407/files/{name}/content"
    with partial.open("ab") as out:
        for a in range(start, n, chunk):
            e = min(a + chunk, n) - 1
            for attempt in range(3):
                try:
                    b = block(url, a, e, n, opener)
                    break
                except urllib.error.HTTPError as err:
                    if err.code not in (500, 502, 503, 504) or attempt == 2:
                        raise
                except (
                    TimeoutError,
                    urllib.error.URLError,
                    http.client.IncompleteRead,
                    ConnectionError,
                ):
                    if attempt == 2:
                        raise
                time.sleep(3 * (attempt + 1))
            out.write(b)
            out.flush()
            os.fsync(out.fileno())
    m, h = hashes(partial)
    need(
        m == md and partial.stat().st_size == n,
        "publisher MD5 mismatch; retaining partial for diagnosis",
    )
    # Hard-link publication never replaces an existing file; verified and partial are on one volume.
    os.link(partial, final)
    partial.unlink()
    print(f"{name}: full publisher checksum verified", flush=True)
    return h


def acquire(folder):
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    s = source()
    with closing(sqlite3.connect(folder / "acquire.sqlite3", timeout=0)) as db:
        db.execute("begin immediate")
        digests = {n: obtain(folder, n, r) for n, r in s["files"].items()}
        need(digests[META] == s["acquisition_metadata_sha256"], "different scanner metadata")
        output = folder / "source-receipt.json"
        b = (
            json.dumps(dict(source_manifest_sha256=SOURCE, full_file_sha256=digests), indent=2)
            + "\n"
        ).encode()
        if output.exists():
            need(output.read_bytes() == b, "different source receipt already exists")
        else:
            with tempfile.TemporaryDirectory(dir=folder) as td:
                p = Path(td) / "receipt"
                p.write_bytes(b)
                os.link(p, output)
        return digests


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder", type=Path)
    acquire(p.parse_args().folder)
