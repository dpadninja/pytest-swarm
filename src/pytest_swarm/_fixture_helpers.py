"""Fixture inspection utilities — pure analysis of FixtureManager metadata."""

from __future__ import annotations

from typing import Any, List

import pytest


#: Built-in pytest fixtures the parallel path can resolve for itself.
#: Everything else shipped by pytest either needs a real FixtureRequest or mutates
#: process-global state (stdout redirection, logging handlers, os.environ), which no
#: amount of per-thread instancing can make safe — such groups run sequentially.
_SUPPORTED_BUILTIN_FIXTURES = frozenset({
    "cache",
    "doctest_namespace",
    "pytestconfig",
    "record_property",
    "tmp_path",
    "tmp_path_factory",
})


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


def _needed_fixtures(item: pytest.Item, fm: Any) -> set:
    """All fixture names *item* transitively needs, args and autouse alike."""
    needed: set = set()
    _collect_deps(item._fixtureinfo.argnames, item, fm, needed)
    _collect_deps(_extra_fixture_names(item), item, fm, needed)
    return needed


def _parametrized_broad_fixtures(items: List[pytest.Item]) -> List[str]:
    """Broad-scope fixtures in the group whose value can differ between items.

    That is any broad-scope fixture which is itself parametrized, plus any that
    transitively depends on a parametrized name. They cannot be pre-fetched once and
    shared; each distinct combination of parameter values needs its own instance.
    """
    fm = items[0].session._fixturemanager

    parametrized: set = set()
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec:
            parametrized.update(callspec.params)
    if not parametrized:
        return []

    ref = items[0]
    tainted: List[str] = []
    for name in _needed_fixtures(ref, fm):
        defs = fm.getfixturedefs(name, ref)
        if not defs or _fixture_scope_name(defs[-1].scope) == "function":
            continue
        if name in parametrized:
            tainted.append(name)
            continue
        deps: set = set()
        _collect_deps(defs[-1].argnames, ref, fm, deps)
        if deps & parametrized:
            tainted.append(name)
    return tainted


def _parallel_blockers(items: List[pytest.Item]) -> List[str]:
    """Reasons the parallel path cannot be used for *items*, empty if it can.

    The only blocker left is an unsupported built-in pytest fixture: one that either
    needs a real FixtureRequest or mutates process-global state. A fixture merely
    named like a built-in but overridden by the user is not a blocker, and neither is
    parametrization — broad-scope fixtures that differ per item get one instance per
    parameter value (see _prefetch_parametrized_broad_scope).
    """
    fm = items[0].session._fixturemanager
    reasons: dict = {}

    for item in items:
        callspec = getattr(item, "callspec", None)
        callspec_params = set(callspec.params) if callspec else set()

        for name in _needed_fixtures(item, fm):
            if name in _SUPPORTED_BUILTIN_FIXTURES or name in callspec_params:
                continue
            defs = fm.getfixturedefs(name, item)
            if not defs:
                continue
            for d in defs:
                func = getattr(d, "func", None)
                if func and func.__module__.startswith("_pytest."):
                    reasons[name] = (
                        f"built-in fixture {name!r} is not supported in worker threads"
                    )
    return [reasons[name] for name in sorted(reasons)]
