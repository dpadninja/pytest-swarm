"""Parallel and serial test execution paths."""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from ._fixture_helpers import (
    _can_run_parallel_setup,
    _extra_fixture_names,
    _fixture_scope_name,
    _is_same_for_all_items,
)
from ._resolver import _MinimalRequest, _drain_generator, _resolve_fixture

# Scope resolution order for sorting pre-fetch dependencies:
# wider scopes must be resolved before narrower ones.
_SCOPE_ORDER = {"session": 0, "package": 1, "module": 2, "class": 3, "function": 4}


# ---------------------------------------------------------------------------
# Shared teardown helpers
# ---------------------------------------------------------------------------

def _teardown_silent(fins: list) -> None:
    """Run *fins* in reverse order, silently swallowing all exceptions."""
    for fn in reversed(fins):
        try:
            fn()
        except Exception:
            pass


def _run_finalizers(fins: list) -> None:
    """Run *fins* in reverse order, re-raising the first exception after all have run."""
    first_exc: BaseException | None = None
    for fn in reversed(fins):
        try:
            fn()
        except Exception as exc:
            if first_exc is None:
                first_exc = exc
    if first_exc is not None:
        raise first_exc


# ---------------------------------------------------------------------------
# Broad-scope fixture cache
# ---------------------------------------------------------------------------

@dataclass
class BroadScopeCache:
    """Shared cache for broad-scope (session / package / module / class) fixtures.

    Fixtures are pre-fetched in the main thread and shared among parallel workers.
    Teardown methods clear the corresponding level and run its finalizers.
    """

    session: dict[str, Any] = field(default_factory=dict)
    session_fin: list = field(default_factory=list)
    package: dict[str, Any] = field(default_factory=dict)
    package_fin: list = field(default_factory=list)
    module: dict[str, Any] = field(default_factory=dict)
    module_fin: list = field(default_factory=list)
    klass: dict[str, Any] = field(default_factory=dict)
    klass_fin: list = field(default_factory=list)

    def merged(self) -> dict[str, Any]:
        """All cached values merged into one dict (narrower scopes win on collision)."""
        return {**self.session, **self.package, **self.module, **self.klass}

    def store(self, scope: str, name: str, value: Any, fins: list) -> None:
        """Store *value* in the bucket that matches *scope* and record its finalizers."""
        if scope == "session":
            self.session[name] = value
            self.session_fin.extend(fins)
        elif scope == "package":
            self.package[name] = value
            self.package_fin.extend(fins)
        elif scope == "module":
            self.module[name] = value
            self.module_fin.extend(fins)
        else:  # class
            self.klass[name] = value
            self.klass_fin.extend(fins)

    def teardown_class(self) -> None:
        _teardown_silent(self.klass_fin)
        self.klass.clear()
        self.klass_fin.clear()

    def teardown_module(self) -> None:
        self.teardown_class()
        _teardown_silent(self.module_fin)
        self.module.clear()
        self.module_fin.clear()

    def teardown_package(self) -> None:
        self.teardown_module()
        _teardown_silent(self.package_fin)
        self.package.clear()
        self.package_fin.clear()

    def teardown_all(self) -> None:
        self.teardown_package()
        _teardown_silent(self.session_fin)
        self.session_fin.clear()


# ---------------------------------------------------------------------------
# Parallel full path: setup + call + teardown per thread
# ---------------------------------------------------------------------------

class _StubSetupState:
    """Disposable SetupState substitute used during marker-hook evaluation.

    Lets skip/xfail hooks populate item.stash without touching the real
    SetupState, which must stay consistent for sequential tests that follow.
    """

    def __init__(self) -> None:
        self.stack: dict = {}

    def setup(self, item: Any) -> None: pass
    def teardown_exact(self, item: Any, nextitem: Any = None) -> None: pass
    def teardown_all(self) -> None: pass
    def _pop_and_teardown(self) -> None: pass
    def _teardown_with_finalization(self, node: Any) -> None: pass


def _prefetch_one(
    name: str,
    ref_item: pytest.Item,
    session: pytest.Session,
    all_broad: dict[str, Any],
    fm: Any,
    cache: BroadScopeCache,
    _from_fd: Any = None,
) -> None:
    """Resolve *name* and any broad-scope fixtures it depends on, storing each
    one into *cache* under its own scope/finalizers as soon as it is computed.

    A fixture reached only *transitively* - as another broad-scope fixture's
    dependency, rather than directly via the outer loop in
    _prefetch_broad_scope - must still get its own cache.store() call. Storing
    depth-first, in true dependency order, as each name is resolved (instead
    of resolving a whole dependency chain and only storing the top-level
    name) guarantees that: relying on the *sort order* of same-scope siblings
    to happen to visit a dependency before its dependent is not reliable,
    since Python's string hash randomization makes that order vary between
    interpreter runs.

    *_from_fd* is set when resolving a same-named dependency of an overriding
    fixture (e.g. `def resource(resource): ...`) - see _resolve_fixture in
    _resolver.py for why this is needed to avoid infinite recursion.
    """
    if name in all_broad:
        return
    defs = fm.getfixturedefs(name, ref_item)
    if not defs:
        return

    if _from_fd is not None and _from_fd in defs:
        idx = defs.index(_from_fd) - 1
        if idx < 0:
            raise LookupError(
                f"Fixture '{name}' requests itself and there is no wider "
                "fixture of the same name to fall back to."
            )
        fd = defs[idx]
    else:
        fd = defs[-1]

    scope = _fixture_scope_name(fd.scope)
    if scope == "function":
        return

    fins: list = []
    kwargs: dict[str, Any] = {}
    for dep in fd.argnames:
        if dep == "request":
            kwargs["request"] = _MinimalRequest(ref_item, fd, all_broad, fins, session)
            continue
        _prefetch_one(
            dep, ref_item, session, all_broad, fm, cache,
            _from_fd=fd if dep == name else None,
        )
        if dep not in all_broad:
            # function-scope dependency, or a plain (non-fixture) value
            _resolve_fixture(dep, ref_item, session, all_broad, fins)
        kwargs[dep] = all_broad[dep]

    result = fd.func(**kwargs)
    if inspect.isgenerator(result):
        value = next(result)
        fins.append(lambda gen=result: _drain_generator(gen))
    else:
        value = result

    all_broad[name] = value
    cache.store(scope, name, value, fins)


def _prefetch_broad_scope(
    items: list[pytest.Item],
    session: pytest.Session,
    cache: BroadScopeCache,
) -> None:
    """Pre-fetch broad-scope fixtures in the main thread and persist them in *cache*.

    Skips fixtures that are already cached or differ across items (indirect params).
    Resolves in scope order (session before module before class) so that
    narrower-scope fixtures that depend on broader ones find their deps ready.

    Only the item's *direct* fixture names seed the loop below - not the full
    transitive closure. A fixture's own dependencies must be resolved by its
    own recursive call in _prefetch_one (which walks fd.argnames in
    declaration order), exactly as real pytest's FixtureDef.execute does.
    Flattening the whole closure into this loop would let leaf dependencies
    that happen to share a scope with their dependent get resolved
    independently, in a tie-broken order that has nothing to do with
    declaration order, before their dependent ever gets a turn.
    """
    fm = session._fixturemanager
    ref_item = items[0]
    all_broad = cache.merged()

    top_level: list[str] = []
    seen: set[str] = set()
    for name in (*ref_item._fixtureinfo.argnames, *_extra_fixture_names(ref_item)):
        if name in seen or name == "request":
            continue
        seen.add(name)
        top_level.append(name)

    def _scope_key(name: str) -> int:
        defs = fm.getfixturedefs(name, ref_item)
        return _SCOPE_ORDER.get(_fixture_scope_name(defs[-1].scope), 99) if defs else 99

    for name in sorted(top_level, key=_scope_key):
        if not _is_same_for_all_items(name, items):
            continue
        _prefetch_one(name, ref_item, session, all_broad, fm, cache)


def _run_one_item(
    item: pytest.Item,
    session: pytest.Session,
    resolved_base: dict[str, Any],
    marker_exc: dict[str, BaseException],
    lock: threading.Lock,
) -> None:
    """Execute one swarm item (setup → call → teardown) in the calling thread.

    *resolved_base* is the pre-fetched broad-scope dict; function-scope fixtures
    are resolved on top of it. Reports are emitted under *lock* via
    pytest_runtest_protocol so hookwrappers fire in the main thread's order.
    """
    from _pytest.runner import CallInfo

    fm = session._fixturemanager
    finalizers: list = []
    resolved: dict[str, Any] = dict(resolved_base)
    reports: list = []

    def _do_setup() -> None:
        exc = marker_exc.get(item.nodeid)
        if exc is not None:
            raise exc
        for name in item._fixtureinfo.argnames:
            if name not in resolved:
                _resolve_fixture(name, item, session, resolved, finalizers)
        for name in _extra_fixture_names(item):
            if name not in resolved:
                defs = fm.getfixturedefs(name, item)
                if defs and _fixture_scope_name(defs[-1].scope) == "function":
                    _resolve_fixture(name, item, session, resolved, finalizers)
        item.funcargs.update(resolved)

    setup_call = CallInfo.from_call(_do_setup, "setup", reraise=(SystemExit, KeyboardInterrupt))
    setup_rep = item.ihook.pytest_runtest_makereport(item=item, call=setup_call)
    reports.append(setup_rep)

    if setup_rep.passed:
        def _do_call() -> None:
            args = {arg: item.funcargs[arg] for arg in item._fixtureinfo.argnames}
            if item.instance is not None:
                item.function(item.instance, **args)
            else:
                item.function(**args)

        call_call = CallInfo.from_call(_do_call, "call", reraise=(SystemExit, KeyboardInterrupt))
        call_rep = item.ihook.pytest_runtest_makereport(item=item, call=call_call)
        reports.append(call_rep)

    teardown_call = CallInfo.from_call(
        lambda: _run_finalizers(finalizers),
        "teardown",
        reraise=(SystemExit, KeyboardInterrupt),
    )
    teardown_rep = item.ihook.pytest_runtest_makereport(item=item, call=teardown_call)
    reports.append(teardown_rep)

    with lock:
        item._swarm_reports = reports  # type: ignore[attr-defined]
        item.ihook.pytest_runtest_protocol(item=item, nextitem=None)
        del item._swarm_reports  # type: ignore[attr-defined]


def _run_items_parallel_full(
    items: list[pytest.Item],
    session: pytest.Session,
    max_workers: int,
    cache: BroadScopeCache,
) -> None:
    """
    Run each item in its own thread: resolve fixtures → call → teardown.

    The main thread's SetupState is never touched. Broad-scope fixtures are
    pre-fetched here and persisted in *cache* for reuse by later groups.
    """
    # Broad-scope fixtures run here, in the main thread, outside of any
    # per-item CallInfo wrapper. A raising fixture must not escape this
    # function - left uncaught it would blow past pytest_runtestloop and
    # crash the whole session with INTERNALERROR instead of failing the
    # affected tests. Stash it and fold it into marker_exc below so it is
    # reported the normal way, once per item, via _do_setup/CallInfo.
    prefetch_exc: BaseException | None = None
    try:
        _prefetch_broad_scope(items, session, cache)
    except BaseException as exc:
        prefetch_exc = exc
    resolved_base = cache.merged()

    # Evaluate marker hooks (skip, xfail, …) with a stub SetupState so they
    # populate item.stash without disturbing the real SetupState.
    marker_exc: dict[str, BaseException] = {}
    orig_ss = session._setupstate
    session._setupstate = _StubSetupState()  # type: ignore[assignment]
    try:
        for item in items:
            try:
                item.ihook.pytest_runtest_setup(item=item)
            except BaseException as exc:
                marker_exc[item.nodeid] = exc
    finally:
        session._setupstate = orig_ss

    if prefetch_exc is not None:
        for item in items:
            marker_exc.setdefault(item.nodeid, prefetch_exc)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(_run_one_item, item, session, resolved_base, marker_exc, lock)
            for item in items
        ]
        for f in futures:
            f.result()


# ---------------------------------------------------------------------------
# Serial setup path: sequential setup, parallel bodies, sequential teardown
# ---------------------------------------------------------------------------

def _run_group_serial_setup(
    items: list[pytest.Item],
    session: pytest.Session,
    max_workers: int,
    nextitem: pytest.Item | None,
) -> None:
    """
    Serial setup → parallel test bodies → serial teardown.

    Used when broad-scope fixtures are indirect-parametrized (different values per
    item) or contain built-in pytest fixtures that require a real FixtureRequest.

    *nextitem* is passed to the teardown of the last item so pytest keeps alive any
    fixtures that the next sequential test still needs (session-scope etc.).
    """
    from _pytest.reports import TestReport
    from _pytest.runner import CallInfo, call_and_report

    setup_reps: list[TestReport] = []
    call_reps: dict[str, Any] = {}
    teardown_reps: list[TestReport] = []
    lock = threading.Lock()

    # Phase 1: serial setup.
    # teardown_exact after each setup removes the item from SetupState.stack without
    # running fixture finalizers — broad-scope fixtures have no functional finalizers
    # here and stay alive in pytest's fixture cache.
    for i, item in enumerate(items):
        rep = call_and_report(item, "setup", log=False)
        setup_reps.append(rep)
        if i < len(items) - 1:
            session._setupstate.teardown_exact(items[i + 1])

    # Phase 2: parallel test bodies.
    def _call_one(item: pytest.Item, setup_rep: TestReport) -> None:
        if not setup_rep.passed:
            return

        def _do_call() -> None:
            args = {arg: item.funcargs[arg] for arg in item._fixtureinfo.argnames}
            if item.instance is not None:
                item.function(item.instance, **args)
            else:
                item.function(**args)

        call_info = CallInfo.from_call(_do_call, "call", reraise=(SystemExit, KeyboardInterrupt))
        rep = item.ihook.pytest_runtest_makereport(item=item, call=call_info)
        with lock:
            call_reps[item.nodeid] = rep

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_call_one, item, setup_reps[i]) for i, item in enumerate(items)]
        for f in futures:
            f.result()

    # Phase 3: teardown.
    # Last item: real teardown (tears down broad-scope fixtures for this group).
    # Others: synthetic no-op teardown reports so makereport hookwrappers fire for
    # every item.
    for i, item in enumerate(items):
        if i < len(items) - 1:
            noop = CallInfo.from_call(lambda: None, "teardown")
            teardown_reps.append(item.ihook.pytest_runtest_makereport(item=item, call=noop))
        else:
            teardown_reps.append(call_and_report(item, "teardown", log=False, nextitem=nextitem))

    # Emit all reports via pytest_runtest_protocol so hookwrappers on it are called.
    for i, item in enumerate(items):
        nid = item.nodeid
        reps: list = [setup_reps[i]]
        if nid in call_reps:
            reps.append(call_reps[nid])
        reps.append(teardown_reps[i])
        item._swarm_reports = reps  # type: ignore[attr-defined]
        item.ihook.pytest_runtest_protocol(item=item, nextitem=None)
        del item._swarm_reports  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_group(
    items: list[pytest.Item],
    session: pytest.Session,
    max_workers: int,
    cache: BroadScopeCache,
    nextitem: pytest.Item | None,
) -> None:
    """Choose and invoke the appropriate execution path for a swarm group.

    Parallel full path: setup + call + teardown entirely per thread; broad-scope
    fixtures pre-fetched in main thread and shared via *cache*.

    Serial setup path: sequential setup, parallel bodies, sequential teardown;
    used when broad-scope fixtures are indirect-parametrized or require a real
    FixtureRequest (e.g. built-in pytest fixtures).
    """
    if _can_run_parallel_setup(items):
        _run_items_parallel_full(items, session, max_workers, cache)
    else:
        _run_group_serial_setup(items, session, max_workers, nextitem)
