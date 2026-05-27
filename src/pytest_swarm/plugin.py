"""
pytest-swarm: run parametrized test variants in parallel threads while respecting
the standard pytest fixture lifecycle.

    @pytest.mark.swarm(max_workers=4)
    @pytest.mark.parametrize("n", range(8))
    def test_heavy(n): ...

Worker count priority: marker max_workers > --swarm-workers > PYTEST_SWARM_WORKERS > cpu_count.
"""

from __future__ import annotations

from ._runner import pytest_addoption, pytest_configure  # noqa: F401
