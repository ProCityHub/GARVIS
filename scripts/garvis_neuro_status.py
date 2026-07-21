#!/usr/bin/env python3
"""Display GARVIS neurocognitive memory statistics."""

from pathlib import Path
import os

from garvis.neurocognitive.store import NeuroStore

home = Path(os.getenv("GARVIS_HOME", str(Path.home() / ".garvis")))
store = NeuroStore(home / "neurocognitive.db")

print("GARVIS neurocognitive memory")
print(f"database: {store.path}")
print(f"memories: {store.count('memories')}")
print(f"episodes: {store.count('episodes')}")
print(f"feedback records: {store.count('feedback')}")
