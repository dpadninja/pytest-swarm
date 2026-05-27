"""SwarmPlugin class and test-loop orchestration."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import pytest

from ._execution import BroadScopeCache, run_group

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
        cli: int | None = config.getoption("swarm_workers")
        env_str = os.environ.get("PYTEST_SWARM_WORKERS")
        env: int | None = int(env_str) if env_str is not None else None
        return cls(global_max=cli if cli is not None else env)


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
# Plugin
# ---------------------------------------------------------------------------

class SwarmPlugin:

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
        all_parallel_nodeids: set[str] = set()
        for item in session.items:
            if item.get_closest_marker(MARKER):
                base = item.nodeid.split("[")[0]
                parallel_groups[base].append(item)
                all_parallel_nodeids.add(item.nodeid)

        cache = BroadScopeCache()
        current_package: str | None = None
        current_module: str | None = None
        current_class: type | None = None
        processed: set[str] = set()

        for i, item in enumerate(session.items):
            if item.nodeid in processed:
                continue

            if item.get_closest_marker(MARKER):
                base = item.nodeid.split("[")[0]
                group = parallel_groups[base]
                for g in group:
                    processed.add(g.nodeid)

                current_package, current_module, current_class = _advance_scope_boundary(
                    item, cache, current_package, current_module, current_class
                )

                marker = item.get_closest_marker(MARKER)
                max_workers = worker_cfg.resolve(marker)
                nextitem = next(
                    (it for it in session.items if it.nodeid not in processed),
                    None,
                )
                run_group(group, session, max_workers, cache, nextitem)
            else:
                # nextitem must point only to the next *sequential* test.
                # Swarm tests do not touch SetupState between groups — passing them
                # as nextitem would incorrectly signal that module/session fixtures
                # are still alive.
                nextitem = next(
                    (session.items[j]
                     for j in range(i + 1, len(session.items))
                     if session.items[j].nodeid not in processed
                     and session.items[j].nodeid not in all_parallel_nodeids),
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
    parser.addoption(
        "--swarm-workers",
        dest="swarm_workers",
        metavar="N",
        type=int,
        default=None,
        help=(
            "Maximum number of worker threads for @pytest.mark.swarm tests. "
            "Overrides PYTEST_SWARM_WORKERS env var. Defaults to CPU count. "
            "Can be overridden per-test via @pytest.mark.swarm(max_workers=N)."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    if not config.pluginmanager.has_plugin("swarm"):
        config.pluginmanager.register(SwarmPlugin(), "swarm")
