"""SwarmPlugin class and test-loop orchestration."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest

from ._execution import BroadScopeCache, run_group
from ._plan import MODE_SEQUENTIAL, GroupPlan, plan_group

MARKER = "swarm"


# ---------------------------------------------------------------------------
# Worker-count configuration
# ---------------------------------------------------------------------------

@dataclass
class WorkerConfig:
    """Resolved global worker limit (CLI or env var).

    Priority: marker max_workers > CLI --swarm-workers > PYTEST_SWARM_WORKERS > cpu_count.
    """

    global_max: int | None

    def resolve(self, marker: Any) -> int:
        """Return max_workers for *marker*, applying the full priority chain."""
        if marker:
            m = marker.kwargs.get("max_workers")
            if m is not None:
                return m
        if self.global_max is not None:
            return self.global_max
        return os.cpu_count() or 1

    @classmethod
    def from_config(cls, config: pytest.Config) -> WorkerConfig:
        return cls(global_max=config.getoption("swarm_workers"))


# ---------------------------------------------------------------------------
# Scope-boundary helper
# ---------------------------------------------------------------------------

def _advance_scope_boundary(
    item: pytest.Item,
    cache: BroadScopeCache,
    current_package: str | None,
    current_module: str | None,
    current_class: type | None,
) -> tuple[str, str, type | None]:
    """Tear down cached fixtures at the appropriate scope when the boundary changes.

    Returns the updated (package, module, class) tracking triple.
    """
    item_package = str(item.path.parent)
    item_module = str(item.path)
    item_class = item.cls

    if item_package != current_package:
        cache.teardown_package()
    elif item_module != current_module:
        cache.teardown_module()
    elif item_class != current_class:
        cache.teardown_class()

    return item_package, item_module, item_class


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

@dataclass
class _GroupReport:
    """What happened to one swarm group, for the --swarm-explain summary."""

    base: str
    count: int
    plan: GroupPlan
    workers: int


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class SwarmPlugin:

    def __init__(self) -> None:
        # One entry per swarm group seen this session, for --swarm-explain.
        self._plans: list[_GroupReport] = []

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_protocol(self, item: pytest.Item, nextitem: pytest.Item | None) -> bool | None:
        reports = getattr(item, "_swarm_reports", None)
        if reports is None:
            return None
        item.ihook.pytest_runtest_logstart(nodeid=item.nodeid, location=item.location)
        for rep in reports:
            item.ihook.pytest_runtest_logreport(report=rep)
        item.ihook.pytest_runtest_logfinish(nodeid=item.nodeid, location=item.location)
        return True

    @pytest.hookimpl
    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        """Report how each swarm group was run.

        Full table under --swarm-explain. Without it, only a one-line notice when a
        group lost its parallelism — that is a silent behaviour change otherwise.
        """
        if not self._plans:
            return

        if terminalreporter.config.getoption("swarm_explain"):
            terminalreporter.write_sep("=", "swarm plan")
            for rep in self._plans:
                workers = f"{rep.workers} worker(s)" if rep.plan.threaded else "no threads"
                terminalreporter.write_line(
                    f"{rep.plan.mode:<10} {rep.count:>3} item(s)  {workers:<12} {rep.base}"
                )
                for reason in rep.plan.reasons:
                    terminalreporter.write_line(f"{'':<12}{reason}")
            return

        demoted = [r for r in self._plans if r.plan.mode == MODE_SEQUENTIAL]
        if demoted:
            terminalreporter.write_line(
                f"swarm: {len(demoted)} group(s) ran sequentially to keep fixtures "
                "alive — use --swarm-explain for details"
            )

    @pytest.hookimpl
    def pytest_configure(self, config: pytest.Config) -> None:
        config.addinivalue_line(
            "markers",
            f"{MARKER}(max_workers=N): run parametrized variants in parallel threads. "
            "max_workers overrides --swarm-workers for this test.",
        )

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session: pytest.Session) -> bool | None:
        if session.config.option.collectonly:
            return None

        worker_cfg = WorkerConfig.from_config(session.config)

        # Build an index of swarm groups: base_nodeid -> [items].
        parallel_groups: dict[str, list[pytest.Item]] = defaultdict(list)
        for item in session.items:
            if item.get_closest_marker(MARKER):
                parallel_groups[item.nodeid.split("[")[0]].append(item)

        # Decide up front how each group can run. A group planned "sequential" is
        # handed back to pytest's ordinary protocol below and is deliberately absent
        # from threaded_nodeids, so it behaves exactly like an unmarked test.
        plans: dict[str, GroupPlan] = {
            base: plan_group(group) for base, group in parallel_groups.items()
        }
        threaded_nodeids: set[str] = {
            it.nodeid
            for base, group in parallel_groups.items()
            if plans[base].threaded
            for it in group
        }
        self._plans = [
            _GroupReport(
                base=base,
                count=len(parallel_groups[base]),
                plan=plans[base],
                workers=(
                    worker_cfg.resolve(parallel_groups[base][0].get_closest_marker(MARKER))
                    if plans[base].threaded
                    else 1
                ),
            )
            for base in parallel_groups
        ]

        cache = BroadScopeCache()
        current_package: str | None = None
        current_module: str | None = None
        current_class: type | None = None
        processed: set[str] = set()

        for i, item in enumerate(session.items):
            if item.nodeid in processed:
                continue

            if item.nodeid in threaded_nodeids:
                base = item.nodeid.split("[")[0]
                group = parallel_groups[base]
                for g in group:
                    processed.add(g.nodeid)

                current_package, current_module, current_class = _advance_scope_boundary(
                    item, cache, current_package, current_module, current_class
                )

                marker = item.get_closest_marker(MARKER)
                max_workers = worker_cfg.resolve(marker)
                run_group(group, session, max_workers, cache, plans[base])
            else:
                # nextitem must point only to the next *sequential* test.
                # Swarm tests do not touch SetupState between groups — passing them
                # as nextitem would incorrectly signal that module/session fixtures
                # are still alive.
                nextitem = next(
                    (session.items[j]
                     for j in range(i + 1, len(session.items))
                     if session.items[j].nodeid not in processed
                     and session.items[j].nodeid not in threaded_nodeids),
                    None,
                )
                item.config.hook.pytest_runtest_protocol(item=item, nextitem=nextitem)

            if session.shouldfail or session.shouldstop:
                break

        cache.teardown_all()
        return True


# ---------------------------------------------------------------------------
# Entry points (re-exported by plugin.py)
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    _env = os.environ.get("PYTEST_SWARM_WORKERS")
    parser.addoption(
        "--swarm-workers",
        dest="swarm_workers",
        metavar="N",
        type=int,
        default=int(_env) if _env is not None else None,
        help=(
            "Maximum number of worker threads for @pytest.mark.swarm tests. "
            "Overrides PYTEST_SWARM_WORKERS env var. Defaults to CPU count. "
            "Can be overridden per-test via @pytest.mark.swarm(max_workers=N)."
        ),
    )
    parser.addoption(
        "--swarm-explain",
        dest="swarm_explain",
        action="store_true",
        default=False,
        help=(
            "After the run, print how each @pytest.mark.swarm group was executed "
            "(parallel / serial / sequential) and why."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.pluginmanager.has_plugin("swarm"):
        config.pluginmanager.register(SwarmPlugin(), "swarm")
