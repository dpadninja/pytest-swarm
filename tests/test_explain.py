"""--swarm-explain reports how each swarm group was executed, and why."""

from __future__ import annotations

import pytest


_PARALLEL = """
import pytest

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("i", range(3))
def test_plain(i):
    pass
"""

_DEMOTED = """
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("i", range(3))
def test_demoted(capsys, i):
    pass
"""


class TestExplainOutput:

    def test_parallel_group_is_listed(self, pytester):
        pytester.makepyfile(_PARALLEL)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["*swarm plan*", "parallel*3 item(s)*4 worker(s)*"])

    def test_sequential_group_reports_its_reason(self, pytester):
        pytester.makepyfile(_DEMOTED)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines([
            "*swarm plan*",
            "sequential*3 item(s)*no threads*",
            "*'capsys' is not supported in worker threads*",
        ])

    def test_no_swarm_tests_prints_nothing(self, pytester):
        pytester.makepyfile("def test_plain(): pass")
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=1)
        assert "swarm plan" not in result.stdout.str()


class TestDefaultNotice:

    def test_demotion_is_announced_without_the_flag(self, pytester):
        """Losing parallelism must never be silent."""
        pytester.makepyfile(_DEMOTED)
        result = pytester.runpytest()
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["*1 group(s) ran sequentially*"])

    def test_quiet_when_everything_ran_in_parallel(self, pytester):
        pytester.makepyfile(_PARALLEL)
        result = pytester.runpytest()
        result.assert_outcomes(passed=3)
        assert "ran sequentially" not in result.stdout.str()
