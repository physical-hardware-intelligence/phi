#!/usr/bin/env python3
"""Check a local Hugging Face download against the Hub's own file listing.

Every listed file must be present, with the listed size and the listed hash:
sha256 for LFS files (the Hub's `lfs.oid`), git blob sha1 for the rest (`oid`).
Files on disk that the listing doesn't name are reported too.

Two steps, so the check also runs where the Hub is unreachable:

    # 1. where the Hub is reachable: save the listing next to the files
    verify_hub_download.py fetch IliaLarchenko/lehome_real 03b26d31bfe6859b438cb793bb911d607226eef6 \
        checkpoints/lehome_real
    verify_hub_download.py fetch lehome/dataset_challenge_real 5a28286fb60db8bf9fab9a552a001645722752cd \
        data/lehome_real --repo-type dataset \
        --prefix top_long_merged top_short_merged pant_long_merged pant_short_merged

    # 2. anywhere the files are, including after copying them to another machine
    verify_hub_download.py check checkpoints/lehome_real

The listing is saved as `.hub_listing.json` inside the download folder, so it travels
with the files. Ignored on disk: `.cache/` (the hf client's own metadata), macOS `._*`
and `.DS_Store`. Standard library only: runs under any python3 >= 3.8.
"""
from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request

LISTING = ".hub_listing.json"
CHUNK = 8 << 20  # 8 MiB reads


def _get_json_pages(url: str) -> list:
    """GET a Hub API listing, following `Link: <...>; rel="next"` pagination."""
    out = []
    while url:
        with urllib.request.urlopen(url, timeout=60) as r:
            out += json.load(r)
            nxt = None
            for part in (r.headers.get("Link") or "").split(","):
                if 'rel="next"' in part:
                    nxt = part[part.find("<") + 1 : part.find(">")]
            url = nxt
    return out


def fetch(args: argparse.Namespace) -> int:
    base = f"https://huggingface.co/api/{args.repo_type}s/{args.repo}/tree/{args.revision}"
    files = []
    for prefix in args.prefix or [""]:
        url = base + (f"/{urllib.parse.quote(prefix)}" if prefix else "") + "?recursive=true"
        for e in _get_json_pages(url):
            if e.get("type") == "file":
                files.append({
                    "path": e["path"],
                    "size": e["size"],
                    "oid": e["oid"],
                    "lfs_sha256": (e.get("lfs") or {}).get("oid"),
                })
    if not files:
        print(f"no files listed for {args.repo}@{args.revision} {args.prefix or ''}", file=sys.stderr)
        return 1
    meta = {
        "repo": args.repo,
        "repo_type": args.repo_type,
        "revision": args.revision,
        "prefixes": args.prefix or [],
        "fetched_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "files": sorted(files, key=lambda f: f["path"]),
    }
    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, LISTING), "w") as fh:
        json.dump(meta, fh, indent=1)
    total = sum(f["size"] for f in files)
    print(f"saved {LISTING}: {len(files)} files, {total:,} bytes, {sum(bool(f['lfs_sha256']) for f in files)} LFS")
    return 0


def _hash(path: str, size: int, lfs: bool) -> str:
    """sha256 of the content (LFS), or git's blob sha1: sha1(b"blob <size>\\0" + content)."""
    h = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % size)
    with open(path, "rb") as fh:
        while block := fh.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _ignored(rel: str) -> bool:
    parts = rel.split(os.sep)
    return parts[0] == ".cache" or rel == LISTING or any(p.startswith("._") or p == ".DS_Store" for p in parts)


def check(args: argparse.Namespace) -> int:
    with open(os.path.join(args.dir, LISTING)) as fh:
        meta = json.load(fh)
    files = meta["files"]
    print(f"{meta['repo']} @ {meta['revision'][:7]} ({meta['repo_type']}) -> {args.dir}")
    bad = 0
    nbytes = 0
    for i, f in enumerate(files, 1):
        p = os.path.join(args.dir, f["path"])
        if not os.path.isfile(p):
            print(f"  MISSING  {f['path']}")
            bad += 1
            continue
        size = os.path.getsize(p)
        if size != f["size"]:
            print(f"  SIZE     {f['path']}: {size:,} != {f['size']:,}")
            bad += 1
            continue
        if not args.no_hash:
            lfs = bool(f["lfs_sha256"])
            want = f["lfs_sha256"] if lfs else f["oid"]
            got = _hash(p, size, lfs)
            if got != want:
                print(f"  HASH     {f['path']}: {got} != {want}")
                bad += 1
                continue
        nbytes += size
        if args.verbose:
            print(f"  ok {i:>3}/{len(files)}  {f['path']}")
    listed = {f["path"] for f in files}
    extras = []
    for root, _, names in os.walk(args.dir):
        for n in names:
            rel = os.path.relpath(os.path.join(root, n), args.dir)
            if rel not in listed and not _ignored(rel) and not any(fnmatch.fnmatch(rel, g) for g in args.allow_extra):
                extras.append(rel)
    for rel in sorted(extras):
        print(f"  EXTRA    {rel}")
    ok = bad == 0 and not extras
    what = "size only" if args.no_hash else "size + hash"
    print(f"  {len(files) - bad}/{len(files)} files match ({what}), {nbytes:,} bytes, "
          f"{len(extras)} unlisted -> {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="save the Hub listing into the download folder")
    f.add_argument("repo")
    f.add_argument("revision", help="full commit sha, so the listing is reproducible")
    f.add_argument("dir")
    f.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    f.add_argument("--prefix", nargs="*", help="only these top-level folders")
    c = sub.add_parser("check", help="verify the folder against its saved listing")
    c.add_argument("dir")
    c.add_argument("--no-hash", action="store_true", help="sizes only (fast)")
    c.add_argument("--allow-extra", nargs="*", default=[], metavar="GLOB",
                   help="files that may sit in the folder without being in the listing (e.g. a repo's own .gitignore)")
    c.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    return fetch(args) if args.cmd == "fetch" else check(args)


if __name__ == "__main__":
    sys.exit(main())
