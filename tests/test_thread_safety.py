from __future__ import annotations

import pytest

from ._helpers import _events, _FAKE_REMOTE_CLASS


def test_no_deadlock_with_global_threaded_resource(pytester):
    """
    A global resource with a background thread (similar to paramiko.Transport)
    must not cause a deadlock when accessed from parallel test threads.
    """
    pytester.makepyfile("""
        import threading, time, pytest

        class _FakeTransport:
            def __init__(self):
                self._lock = threading.Lock()
                self._thread = threading.Thread(
                    target=lambda: [time.sleep(0.05) for _ in iter(int, 1)],
                    daemon=True,
                )
                self._thread.start()

            def run(self, cmd: str) -> str:
                with self._lock:
                    return f"ok:{cmd}"

        TRANSPORT = _FakeTransport()

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("cmd", ["a", "b", "c", "d"])
        def test_uses_global_transport(cmd):
            result = TRANSPORT.run(cmd)
            assert result == f"ok:{cmd}"
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)


def test_module_scoped_connection_pool(pytester):
    """
    module-scoped connection pool with threading.Lock: setup and teardown must
    each occur exactly once.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makepyfile(f"""
        import threading, pytest

        class _FakePool:
            SIZE = 10

            def __init__(self):
                self._slots = [threading.Lock() for _ in range(self.SIZE)]
                self._total_lock = threading.Lock()
                self.total_calls = 0
                threading.Thread(target=lambda: None, daemon=True).start()

            def run(self, cmd: str) -> str:
                for slot in self._slots:
                    if slot.acquire(blocking=False):
                        try:
                            with self._total_lock:
                                self.total_calls += 1
                            return f"result:{{cmd}}"
                        finally:
                            slot.release()
                raise RuntimeError("no free slots")

        @pytest.fixture(scope="module")
        def pool():
            open({counter!r}, "a").write("SETUP\\n")
            p = _FakePool()
            yield p
            open({counter!r}, "a").write("TEARDOWN\\n")

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("cmd", ["x", "y", "z", "w"])
        def test_uses_pool(cmd, pool):
            result = pool.run(cmd)
            assert result == f"result:{{cmd}}"
            assert pool.total_calls > 0
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert ev.count("SETUP") == 1
    assert ev.count("TEARDOWN") == 1


def test_thread_local_factory_function_scope(pytester):
    """
    threading.local() factory with function-scope: each unique thread creates
    its own connection.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makepyfile(_FAKE_REMOTE_CLASS + f"""
import pytest

COUNTER = {counter!r}

@pytest.fixture
def remote():
    return _FakeRemote("host-a", COUNTER)

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_uses_remote(n, remote):
    assert remote.host == "host-a"
""")
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert 1 <= ev.count("CONNECT") <= 4


def test_thread_local_factory_module_scope(pytester):
    """
    threading.local() factory with module-scope: the connection is reused
    across all parallel variants, so CONNECT fires exactly once.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makepyfile(_FAKE_REMOTE_CLASS + f"""
import pytest

COUNTER = {counter!r}

@pytest.fixture(scope="module")
def remote():
    return _FakeRemote("host-b", COUNTER)

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_uses_remote(n, remote):
    assert remote.host == "host-b"
""")
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert ev.count("CONNECT") == 1
