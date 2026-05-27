from __future__ import annotations

from pathlib import Path


def _events(path: str) -> list[str]:
    try:
        return Path(path).read_text().strip().splitlines()
    except FileNotFoundError:
        return []


_FAKE_REMOTE_CLASS = """
import threading

class _FakeRemote:
    _local = threading.local()

    def __new__(cls, host: str, counter_file: str):
        if not hasattr(cls._local, "cache"):
            cls._local.cache = {}
        key = (host, counter_file)
        if key not in cls._local.cache:
            open(counter_file, "a").write("CONNECT\\n")
            obj = object.__new__(cls)
            obj.host = host
            cls._local.cache[key] = obj
        return cls._local.cache[key]
"""