"""Move the passage index and run results between this machine and the paper library's bucket.

The papers themselves already live in the bucket (paper-fetch's S3 store). The passage index is a
cache built from them, so it travels one way: the Mac uploads a consistent snapshot before a cloud
batch, and the batch server downloads it. Papers the batch fetches are written to the bucket by
paper-fetch as usual, and the Mac indexes them afterwards; no index file is ever copied back over
one that running servers have open.

Credentials and the bucket come from paper-fetch's own settings (PAPER_FETCH_S3_*), read from the
environment or ~/.config/paper-fetch/*.env. Every object is private (no ACL is ever set).

    python cloud/index_sync.py snapshot SRC.sqlite DEST.sqlite   # consistent copy of a live DB
    python cloud/index_sync.py upload FILE KEY                   # skipped if the bucket has it
    python cloud/index_sync.py download KEY DEST
    python cloud/index_sync.py put-dir DIR PREFIX
    python cloud/index_sync.py get-dir PREFIX DIR
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any

ENV_DIR = Path(os.environ.get("PAPER_FETCH_ENV_DIR") or Path.home() / ".config/paper-fetch")


def _load_env() -> None:
    """paper-fetch's env files, without overriding the environment (as paper-fetch does)."""
    if not ENV_DIR.is_dir():
        return
    for path in sorted(ENV_DIR.glob("*.env")):
        for raw in path.read_text().splitlines():
            key, sep, value = raw.strip().partition("=")
            if sep and key.startswith("PAPER_FETCH_") and key not in os.environ:
                os.environ[key] = value.strip().strip("'\"")


def _bucket() -> tuple[Any, str]:
    import boto3  # paper-fetch's s3 extra

    _load_env()
    env = os.environ
    client = boto3.client(
        "s3",
        region_name=env.get("PAPER_FETCH_S3_REGION") or None,
        endpoint_url=env.get("PAPER_FETCH_S3_ENDPOINT_URL") or None,
        aws_access_key_id=env["PAPER_FETCH_S3_ACCESS_KEY_ID"],
        aws_secret_access_key=env["PAPER_FETCH_S3_SECRET_ACCESS_KEY"],
    )
    return client, env["PAPER_FETCH_S3_BUCKET"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def snapshot(src: Path, dest: Path) -> None:
    """A consistent copy of a database other processes may be writing (WAL included)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.unlink(missing_ok=True)
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as source, sqlite3.connect(dest) as out:
        source.backup(out)
    print(f"snapshot {src} -> {dest} ({dest.stat().st_size / 1e9:.2f} GB)")


def upload(path: Path, key: str) -> str:
    client, bucket = _bucket()
    digest = sha256(path)
    try:
        head = client.head_object(Bucket=bucket, Key=key)
        if head.get("Metadata", {}).get("sha256") == digest:
            print(f"unchanged, not uploaded: {key}")
            return digest
    except client.exceptions.ClientError:
        pass
    client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
    print(f"uploaded {key} ({path.stat().st_size / 1e9:.2f} GB, sha256 {digest[:12]})")
    return digest


def download(key: str, dest: Path) -> None:
    client, bucket = _bucket()
    dest.parent.mkdir(parents=True, exist_ok=True)
    expected = client.head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("sha256")
    client.download_file(bucket, key, str(dest))
    if expected and sha256(dest) != expected:
        raise SystemExit(f"{key}: sha256 mismatch after download")
    print(f"downloaded {key} -> {dest}")


def put_dir(directory: Path, prefix: str) -> None:
    client, bucket = _bucket()
    files = [p for p in directory.rglob("*") if p.is_file()]
    for path in files:
        client.upload_file(str(path), bucket, f"{prefix.rstrip('/')}/{path.relative_to(directory)}")
    print(f"uploaded {len(files)} file(s) to {prefix}")


def get_dir(prefix: str, directory: Path) -> None:
    client, bucket = _bucket()
    prefix = prefix.rstrip("/") + "/"
    count = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            dest = directory / item["Key"][len(prefix) :]
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, item["Key"], str(dest))
            count += 1
    print(f"downloaded {count} file(s) from {prefix}")


def main(argv: list[str]) -> int:
    commands = {
        "snapshot": lambda a: snapshot(Path(a[0]), Path(a[1])),
        "upload": lambda a: upload(Path(a[0]), a[1]),
        "download": lambda a: download(a[0], Path(a[1])),
        "put-dir": lambda a: put_dir(Path(a[0]), a[1]),
        "get-dir": lambda a: get_dir(a[0], Path(a[1])),
    }
    if len(argv) != 3 or argv[0] not in commands:
        print(__doc__, file=sys.stderr)
        return 2
    commands[argv[0]](argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
