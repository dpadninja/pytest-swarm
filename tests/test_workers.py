from __future__ import annotations

import time

import pytest

_N = 4
_SLEEP = 0.3


def test_collect_only_does_not_run_tests(pytester):
    """--collect-only must not execute test bodies."""
    pytester.makepyfile("""
        import pytest

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", range(4))
        def test_thing(n):
            raise RuntimeError("should not run during --collect-only")
    """)
    result = pytester.runpytest("--collect-only")
    result.assert_outcomes()


def test_env_var_sets_workers(pytester, monkeypatch):
    """PYTEST_SWARM_WORKERS=1 forces sequential execution."""
    monkeypatch.setenv("PYTEST_SWARM_WORKERS", "1")
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.mark.swarm
        @pytest.mark.parametrize("n", range({_N}))
        def test_item(n):
            time.sleep({_SLEEP})
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    # 1 worker → sequential execution ≈ N * sleep
    assert elapsed >= _N * _SLEEP * 0.8, (
        f"With PYTEST_SWARM_WORKERS=1 expected sequential execution, "
        f"elapsed={elapsed:.2f}s"
    )


def test_cli_overrides_env_var(pytester, monkeypatch):
    """--swarm-workers takes priority over PYTEST_SWARM_WORKERS."""
    monkeypatch.setenv("PYTEST_SWARM_WORKERS", "1")  # env limits to 1 worker
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.mark.swarm
        @pytest.mark.parametrize("n", range({_N}))
        def test_item(n):
            time.sleep({_SLEEP})
    """)
    # CLI sets N workers → parallel execution despite env=1
    t0 = time.monotonic()
    res = pytester.runpytest("-v", f"--swarm-workers={_N}")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    assert elapsed < _N * _SLEEP * 0.6, (
        f"CLI --swarm-workers={_N} should override PYTEST_SWARM_WORKERS=1, "
        f"elapsed={elapsed:.2f}s"
    )
