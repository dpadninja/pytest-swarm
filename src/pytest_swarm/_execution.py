"""Parallel and serial test execution paths."""

from __future__ import annotations

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import pytest

from ._fixture_helpers import (
    _collect_deps,
    _extra_fixture_names,
    _fixture_scope_name,
    _parametrized_broad_fixtures,
)
from ._plan import MODE_SEQUENTIAL, GroupPlan
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

    def add_finalizers(self, scope: str, fins: list) -> None:
        """Record finalizers at *scope* without publishing a shared value.

        Used for parametrized broad-scope fixtures: several instances of the same
        name are alive at once, so no single one of them can own the cache slot,
        but every one of them must still be torn down at the right boundary.
        """
        if scope == "session":
            self.session_fin.extend(fins)
        elif scope == "package":
            self.package_fin.extend(fins)
        elif scope == "module":
            self.module_fin.extend(fins)
        else:  # class
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
    store: Any,
    _from_fd: Any = None,
) -> None:
    """Resolve *name* and any broad-scope fixtures it depends on, storing each
    one via *store* under its own scope/finalizers as soon as it is computed.

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
            dep, ref_item, session, all_broad, fm, store,
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
    store(scope, name, value, fins)


def _scope_key(name: str, ref_item: pytest.Item, fm: Any) -> int:
    """Sort key placing wider scopes before narrower ones."""
    defs = fm.getfixturedefs(name, ref_item)
    return _SCOPE_ORDER.get(_fixture_scope_name(defs[-1].scope), 99) if defs else 99


def _prefetch_broad_scope(
    items: list[pytest.Item],
    session: pytest.Session,
    cache: BroadScopeCache,
) -> dict[str, Any]:
    """Pre-fetch broad-scope fixtures in the main thread and persist them in *cache*.

    Skips fixtures that are already cached or differ across items (parametrized ones
    are resolved per parameter value by _prefetch_parametrized_broad_scope).
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
    parametrized = set(_parametrized_broad_fixtures(items))

    top_level: list[str] = []
    seen: set[str] = set()
    for name in (*ref_item._fixtureinfo.argnames, *_extra_fixture_names(ref_item)):
        if name in seen or name == "request":
            continue
        seen.add(name)
        top_level.append(name)

    for name in sorted(top_level, key=lambda n: _scope_key(n, ref_item, fm)):
        if name in parametrized:
            continue  # handled per parameter value, see _prefetch_parametrized_broad_scope
        _prefetch_one(name, ref_item, session, all_broad, fm, cache.store)

    return all_broad


def _same_value(a: Any, b: Any) -> bool:
    """Equality that never raises — parameter values can have exotic __eq__."""
    try:
        return bool(a == b)
    except Exception:
        return a is b


def _relevant_param_names(
    tainted: list[str], ref_item: pytest.Item, fm: Any, parametrized: set
) -> list[str]:
    """Parameter names whose values decide which instances a *tainted* fixture needs."""
    relevant: set = set()
    for name in tainted:
        if name in parametrized:
            relevant.add(name)
        defs = fm.getfixturedefs(name, ref_item)
        if not defs:
            continue
        deps: set = set()
        _collect_deps(defs[-1].argnames, ref_item, fm, deps)
        relevant.update(deps & parametrized)
    return sorted(relevant)


def _prefetch_parametrized_broad_scope(
    items: list[pytest.Item],
    session: pytest.Session,
    cache: BroadScopeCache,
    all_broad: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Resolve broad-scope fixtures that differ per item — one instance per value.

    This is the thing pytest itself cannot do. A FixtureDef holds a single live
    value, so pytest must tear down the instance for parameter N before it can build
    the one for N+1; every item but the last would end up holding a finalized object
    once the bodies run in parallel. Resolving here, in the main thread, with one
    instance per distinct combination of parameter values, is what lets
    indirect-parametrized broad-scope fixtures take the parallel path at all.

    Returns nodeid -> {fixture name: value}, an overlay applied on top of the shared
    cache for each item. Finalizers go to *cache* under the fixture's own scope, so
    all live instances are torn down together at the right boundary.
    """
    tainted = _parametrized_broad_fixtures(items)
    if not tainted:
        return {}

    fm = session._fixturemanager
    ref_item = items[0]
    parametrized: set = set()
    for item in items:
        callspec = getattr(item, "callspec", None)
        if callspec:
            parametrized.update(callspec.params)

    relevant = _relevant_param_names(tainted, ref_item, fm, parametrized)
    tainted = sorted(tainted, key=lambda n: _scope_key(n, ref_item, fm))

    def _signature(item: pytest.Item) -> list:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        return [params.get(name) for name in relevant]

    # Bucket items into equivalence classes: same parameter values -> same instances.
    buckets: list[tuple[list, list[pytest.Item]]] = []
    for item in items:
        signature = _signature(item)
        for known, members in buckets:
            if len(known) == len(signature) and all(
                _same_value(a, b) for a, b in zip(known, signature)
            ):
                members.append(item)
                break
        else:
            buckets.append((signature, [item]))

    overlay: dict[str, dict[str, Any]] = {}
    for _, members in buckets:
        rep = members[0]
        local = dict(all_broad)

        def _store(scope: str, name: str, value: Any, fins: list) -> None:
            cache.add_finalizers(scope, fins)

        for name in tainted:
            _prefetch_one(name, rep, session, local, fm, _store)

        values = {name: local[name] for name in tainted if name in local}
        for member in members:
            overlay[member.nodeid] = values

    return overlay


#: Built-in fixtures that need touching in the main thread before workers start.
#: tmp_path_factory creates its base directory lazily and is not thread-safe about it.
_BUILTIN_WARMUPS = {
    "tmp_path_factory": lambda value: value.getbasetemp(),
}


def _warm_builtins(resolved_base: dict[str, Any]) -> None:
    """Force lazy initialization of built-in fixtures while still single-threaded."""
    for name, warm in _BUILTIN_WARMUPS.items():
        if name in resolved_base:
            warm(resolved_base[name])


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
    overlay: dict[str, dict[str, Any]] = {}
    try:
        all_broad = _prefetch_broad_scope(items, session, cache)
        _warm_builtins(all_broad)
        overlay = _prefetch_parametrized_broad_scope(items, session, cache, all_broad)
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
            pool.submit(
                _run_one_item,
                item,
                session,
                {**resolved_base, **overlay.get(item.nodeid, {})},
                marker_exc,
                lock,
            )
            for item in items
        ]
        for f in futures:
            f.result()


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def run_group(
    items: list[pytest.Item],
    session: pytest.Session,
    max_workers: int,
    cache: BroadScopeCache,
    plan: GroupPlan,
) -> None:
    """Run a swarm group in worker threads.

    Setup, call and teardown all happen per thread; broad-scope fixtures are
    pre-fetched in the main thread and shared via *cache*, except parametrized ones,
    which get one instance per parameter value.

    Groups planned MODE_SEQUENTIAL never reach here — the runner leaves them to
    pytest's ordinary protocol.
    """
    assert plan.mode != MODE_SEQUENTIAL, "sequential groups are run by the main loop"
    _run_items_parallel_full(items, session, max_workers, cache)
