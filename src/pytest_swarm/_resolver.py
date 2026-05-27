"""Parallel fixture resolver — bypasses pytest SetupState for worker-thread execution."""

from __future__ import annotations

import inspect
from typing import Any

import pytest


class _MinimalRequest:
    """Minimal FixtureRequest stand-in for calling fixture functions inside worker threads."""

    def __init__(
        self,
        item: pytest.Item,
        fd: Any,
        resolved: dict[str, Any],
        finalizers: list,
        session: pytest.Session,
    ) -> None:
        self._item = item
        self._fd = fd
        self._resolved = resolved
        self._finalizers = finalizers
        self._session = session

        self.node = item
        self.config = item.config
        self.session = session
        self.fixturenames = list(item.fixturenames)
        self.scope = "function"
        self.module = getattr(item, "module", None)
        self.cls = getattr(item, "cls", None)
        self.function = getattr(item, "function", None)
        self.keywords = item.keywords
        self.fspath = getattr(item, "fspath", None)

    @property
    def param(self) -> Any:
        callspec = getattr(self._item, "callspec", None)
        if callspec is not None and self._fd.argname in callspec.params:
            return callspec.params[self._fd.argname]
        raise AttributeError(
            f"Fixture '{self._fd.argname}' does not use a 'params' argument "
            "and is not indirectly parametrized"
        )

    def addfinalizer(self, fn: Any) -> None:
        self._finalizers.append(fn)

    def getfixturevalue(self, name: str) -> Any:
        if name not in self._resolved:
            _resolve_fixture(name, self._item, self._session, self._resolved, self._finalizers)
        return self._resolved[name]

    def applymarker(self, marker: Any) -> None:
        self._item.add_marker(marker)


def _drain_generator(gen: Any) -> None:
    try:
        next(gen)
    except StopIteration:
        pass


def _resolve_fixture(
    name: str,
    item: pytest.Item,
    session: pytest.Session,
    resolved: dict[str, Any],
    finalizers: list,
) -> Any:
    """Recursively resolve *name* and its dependencies in the current thread.

    Resolved values accumulate in *resolved*; teardown callables are appended to
    *finalizers* in setup order (caller must run them reversed).
    """
    if name in resolved:
        return resolved[name]
    if name == "request":
        return None  # injected as _MinimalRequest at each call site, not resolved here

    fm = session._fixturemanager
    defs = fm.getfixturedefs(name, item)

    if not defs:
        # Plain @parametrize value — not a fixture, read directly from callspec.
        callspec = getattr(item, "callspec", None)
        if callspec and name in callspec.params:
            resolved[name] = callspec.params[name]
        return resolved.get(name)

    fd = defs[-1]
    kwargs: dict[str, Any] = {}
    for dep in fd.argnames:
        if dep == "request":
            kwargs["request"] = _MinimalRequest(item, fd, resolved, finalizers, session)
        else:
            kwargs[dep] = _resolve_fixture(dep, item, session, resolved, finalizers)

    result = fd.func(**kwargs)
    if inspect.isgenerator(result):
        value = next(result)
        finalizers.append(lambda gen=result: _drain_generator(gen))
    else:
        value = result

    resolved[name] = value
    return value
