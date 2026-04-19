"""Bring up local Qdrant via docker-compose and create the default collections."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("[setup] docker compose up -d qdrant grobid")
    r = subprocess.run(
        ["docker", "compose", "up", "-d", "qdrant", "grobid"],
        cwd=_ROOT,
        check=False,
    )
    if r.returncode != 0:
        print("[setup] docker compose failed — is Docker Desktop running?", file=sys.stderr)
        return r.returncode

    # Create default collections
    from scholarpeer.index.qdrant_client import QdrantStore

    store = QdrantStore()
    store.ensure_hybrid_collection()
    store.ensure_colpali_collection()
    print("[setup] Collections ready: sp_dense, sp_colpali")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
