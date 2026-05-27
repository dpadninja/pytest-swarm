"""Fixture inspection utilities — pure analysis of FixtureManager metadata."""

from __future__ import annotations

from typing import Any

import pytest


def _fixture_scope_name(scope_raw: Any) -> str:
    if hasattr(scope_raw, "name"):
        return scope_raw.name.lower()
    return str(scope_raw).lower()


def _collect_deps(names: list[str], item: pytest.Item, fm: Any, result: set[str]) -> None:
    """Recursively collect all fixture names transitively required by *names*."""
    for name in names:
        if name in result or name == "request":
            continue
        result.add(name)
        defs = fm.getfixturedefs(name, item)
        if defs:
            _collect_deps(defs[-1].argnames, item, fm, result)


def _extra_fixture_names(item: pytest.Item) -> list[str]:
    """Fixtures in item.fixturenames not listed as function args (autouse, usefixtures)."""
    argnames = set(item._fixtureinfo.argnames)
    return [name for name in item.fixturenames if name not in argnames and name != "request"]


def _is_same_for_all_items(name: str, items: list[pytest.Item]) -> bool:
    """True if *name* is not indirect-parametrized — same fixture value for every item."""
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec and name in callspec.params:
            return False
    return True


def _can_run_parallel_setup(items: list[pytest.Item]) -> bool:
    """
    True when every item in the group can run with fully parallel setup in worker threads.

    Returns False in two cases:
    - A built-in pytest fixture (tmp_path, capfd, …) is needed — it requires a real
      FixtureRequest that _MinimalRequest cannot satisfy.
    - A broad-scope fixture has different values per item (indirect parametrization) —
      it must be initialised sequentially in the serial path.

    Broad-scope fixtures that are identical for all items are pre-fetched in the main
    thread and shared, so they do not block the parallel path.
    """
    fm = items[0].session._fixturemanager
    for item in items:
        callspec = getattr(item, "callspec", None)
        callspec_params = set(callspec.params) if callspec else set()

        needed: set[str] = set()
        _collect_deps(item._fixtureinfo.argnames, item, fm, needed)
        _collect_deps(_extra_fixture_names(item), item, fm, needed)

        for name in needed:
            defs = fm.getfixturedefs(name, item)
            if not defs:
                continue
            for d in defs:
                func = getattr(d, "func", None)
                if func and func.__module__.startswith("_pytest.") and name not in callspec_params:
                    return False
                if _fixture_scope_name(d.scope) != "function" and not _is_same_for_all_items(name, items):
                    return False
    return True