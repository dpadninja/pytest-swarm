"""Execution-mode planning for a swarm group.

Two honest outcomes, decided before a group runs:

* ``parallel``   - every item gets its own fixture instances in its own thread.
* ``sequential`` - no parallelism at all; the group runs through pytest's ordinary
  protocol because something it needs cannot be made thread-safe.

There is deliberately no middle mode. A half-parallel path — setting every item up
first, then running the bodies together — cannot preserve fixture lifetimes, because
pytest holds a single live value per FixtureDef and sweeps function-scoped fixtures
between setups. It would take the cost of parallelism without its guarantees, so a
group either parallelizes properly or does not parallelize at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pytest

from ._fixture_helpers import _parallel_blockers

MODE_PARALLEL = "parallel"
MODE_SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class GroupPlan:
    """How a group will be run, and why."""

    mode: str
    reasons: tuple

    @property
    def threaded(self) -> bool:
        """True if the group runs in worker threads."""
        return self.mode == MODE_PARALLEL


def plan_group(items: List[pytest.Item]) -> GroupPlan:
    """Decide how *items* can be run without breaking fixture lifetimes."""
    blockers = _parallel_blockers(items)
    if blockers:
        return GroupPlan(MODE_SEQUENTIAL, tuple(blockers))
    return GroupPlan(MODE_PARALLEL, ())
