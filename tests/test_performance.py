from __future__ import annotations

import time

import pytest

_SLEEP = 1.0
_N = 4


def test_parallel_speedup(pytester):
    """
    Running N tests with sleep(T) in parallel must be at least 2× faster than
    running them sequentially.
    """
    pytester.makepyfile(test_sequential=f"""
        import time, pytest

        @pytest.mark.parametrize("n", range({_N}))
        def test_slow(n):
            time.sleep({_SLEEP})
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("test_sequential.py", "-v")
    seq_time = time.monotonic() - t0
    res.assert_outcomes(passed=_N)

    pytester.makepyfile(test_parallel=f"""
        import time, pytest

        @pytest.mark.swarm(max_workers={_N})
        @pytest.mark.parametrize("n", range({_N}))
        def test_slow(n):
            time.sleep({_SLEEP})
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("test_parallel.py", "-v")
    par_time = time.monotonic() - t0
    res.assert_outcomes(passed=_N)

    speedup = seq_time / par_time
    assert speedup >= 2.0, (
        f"Expected speedup ≥ 2×, got {speedup:.2f}× "
        f"(sequential={seq_time:.2f}s, parallel={par_time:.2f}s)"
    )


def test_fixture_setup_speedup(pytester):
    """
    sleep(T) inside a function-scope fixture setup: each item runs in its own
    thread, so fixture setup executes in parallel.
    """
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.fixture(params=range({_N}))
        def slow_fixture(request):
            time.sleep({_SLEEP})
            yield request.param

        @pytest.mark.swarm(max_workers={_N})
        def test_slow(slow_fixture):
            pass
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    assert elapsed < _N * _SLEEP * 0.6, (
        f"Expected parallel fixture setup (< {_N * _SLEEP * 0.6:.1f}s), "
        f"elapsed={elapsed:.2f}s"
    )


def test_mixed_scope_setup_speedup(pytester):
    """
    module-scope fixture (sleep 2 s, once) + function-scope fixture with params
    (sleep 1 s, N variants in parallel). Expected total ≈ 2 + 1 = 3 s, not 2 + N×1 = 6 s.
    """
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.fixture(scope="module")
        def shared():
            time.sleep(2.0)
            yield {{"ready": True}}

        @pytest.fixture(params=range({_N}))
        def worker(request):
            time.sleep({_SLEEP})
            yield request.param

        @pytest.mark.swarm(max_workers={_N})
        def test_thing(shared, worker):
            assert shared["ready"]
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    assert elapsed < 4.5, (
        f"Expected ~3 s (broad-scope resolved once + function-scope parallel), "
        f"elapsed={elapsed:.2f}s"
    )


# ---------------------------------------------------------------------------
# Cases that used to be demoted off the parallel path
# ---------------------------------------------------------------------------

def test_builtin_fixture_no_longer_serializes_setup(pytester):
    """
    A supported built-in fixture (tmp_path) in the dependency chain used to demote the
    whole group to serial setup: N × T. Function-scope setup now runs per thread, so
    the group costs ~T.
    """
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.fixture
        def slow_setup(tmp_path):
            time.sleep({_SLEEP})
            return tmp_path

        @pytest.mark.swarm(max_workers={_N})
        @pytest.mark.parametrize("n", range({_N}))
        def test_thing(slow_setup, n):
            assert slow_setup.exists()
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    assert elapsed < _N * _SLEEP * 0.6, (
        f"Expected parallel setup (< {_N * _SLEEP * 0.6:.1f}s), not serial "
        f"({_N * _SLEEP:.1f}s); elapsed={elapsed:.2f}s"
    )


def test_indirect_broad_scope_no_longer_serializes_setup(pytester):
    """
    Same for an indirect-parametrized broad-scope fixture. The broad-scope instances
    themselves are still built once each in the main thread — what parallelizes here
    is the function-scope fixture hanging off them.
    """
    pytester.makepyfile(f"""
        import time, pytest

        @pytest.fixture(scope="module")
        def conn(request):
            return "c%s" % request.param

        @pytest.fixture
        def slow_setup(conn):
            time.sleep({_SLEEP})
            return conn

        @pytest.mark.swarm(max_workers={_N})
        @pytest.mark.parametrize("conn", list(range({_N})), indirect=True)
        def test_thing(slow_setup, conn):
            assert slow_setup == conn
    """)
    t0 = time.monotonic()
    res = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    res.assert_outcomes(passed=_N)
    assert elapsed < _N * _SLEEP * 0.6, (
        f"Expected parallel setup (< {_N * _SLEEP * 0.6:.1f}s), not serial "
        f"({_N * _SLEEP:.1f}s); elapsed={elapsed:.2f}s"
    )
