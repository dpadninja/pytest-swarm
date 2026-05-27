from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# pytest_runtest_protocol: hookwrapper and regular hookimpl behaviour
# ---------------------------------------------------------------------------

def test_runtest_protocol_hookwrapper_called(pytester, tmp_path):
    """A hookwrapper on pytest_runtest_protocol must be called for swarm tests."""
    log_file = tmp_path / "protocol_calls.txt"
    pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    with open({str(log_file)!r}, "a") as f:
        f.write(item.nodeid + "\\n")
    yield
""")
    pytester.makepyfile("""
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
def test_items(n):
    pass
""")
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=3)
    calls = log_file.read_text().splitlines()
    assert len(calls) == 3, f"expected 3 pytest_runtest_protocol calls, got {len(calls)}: {calls}"


def test_runtest_protocol_regular_impl_not_called_for_swarm(pytester, tmp_path):
    """A regular @pytest.hookimpl (without hookwrapper) on pytest_runtest_protocol
    is NOT called for swarm tests — this is an intentional limitation.

    pytest_runtest_protocol is a firstresult hook: SwarmPlugin returns True
    (tryfirst), stopping the chain and preventing the standard pytest runner
    from executing the test again via SetupState. Regular implementations
    registered before SwarmPlugin (i.e. in conftest) sit lower in the hookimpl
    list and are never reached.

    Use hookwrapper=True to observe pytest_runtest_protocol.
    """
    log_file = tmp_path / "calls.txt"
    pytester.makeconftest(f"""
import pytest

@pytest.hookimpl          # regular, no hookwrapper
def pytest_runtest_protocol(item, nextitem):
    with open({str(log_file)!r}, "a") as f:
        f.write(item.nodeid + "\\n")
    return None           # explicitly not intercepting

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_logstart(nodeid, location):
    with open({str(log_file)!r}, "a") as f:
        f.write("LOGSTART:" + nodeid + "\\n")
    yield
""")
    pytester.makepyfile("""
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
def test_items(n):
    pass
""")
    result = pytester.runpytest()
    result.assert_outcomes(passed=3)
    calls = log_file.read_text().splitlines() if log_file.exists() else []
    protocol_calls = [c for c in calls if "LOGSTART:" not in c]
    logstart_calls = [c for c in calls if "LOGSTART:" in c]
    # logstart hookwrapper fires; regular protocol hookimpl does not
    assert len(logstart_calls) == 3, f"expected 3 logstart calls, got {logstart_calls}"
    assert len(protocol_calls) == 0, (
        f"expected 0 regular pytest_runtest_protocol calls for swarm tests, "
        f"got {len(protocol_calls)}: {protocol_calls}"
    )


# ---------------------------------------------------------------------------
# Public pytest hooks: compatibility with third-party plugins
# ---------------------------------------------------------------------------

class TestPublicHooks:
    """Public pytest hooks must fire for swarm tests — otherwise third-party
    plugins (reporters, xfail, flaky, …) will not work correctly."""

    @staticmethod
    def _parallel_src() -> str:
        """Test file without broad-scope fixtures → parallel full path."""
        return """
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
def test_items(n):
    pass
"""

    @staticmethod
    def _serial_src() -> str:
        """Test file with a session-scope fixture → serial-setup path."""
        return """
import pytest

@pytest.fixture(scope="session")
def sess():
    return 42

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
def test_items(n, sess):
    assert sess == 42
"""

    @staticmethod
    def _indirect_src() -> str:
        """Indirect parametrize → _can_run_parallel_setup=False → serial-setup path."""
        return """
import pytest

@pytest.fixture(scope="module")
def mod_fix(request):
    return request.param

@pytest.mark.swarm
@pytest.mark.parametrize("mod_fix", ["a", "b", "c"], indirect=True)
def test_items(mod_fix):
    assert mod_fix in ("a", "b", "c")
"""

    @staticmethod
    def _read(path) -> list[str]:
        return path.read_text().splitlines() if path.exists() else []

    # -----------------------------------------------------------------------
    # pytest_runtest_logstart
    # -----------------------------------------------------------------------

    def test_logstart_parallel(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logstart(nodeid, location):
    with open({str(log)!r}, "a") as f:
        f.write(nodeid + "\\n")
""")
        pytester.makepyfile(self._parallel_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    def test_logstart_serial(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logstart(nodeid, location):
    with open({str(log)!r}, "a") as f:
        f.write(nodeid + "\\n")
""")
        pytester.makepyfile(self._serial_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    # -----------------------------------------------------------------------
    # pytest_runtest_logfinish
    # -----------------------------------------------------------------------

    def test_logfinish_parallel(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logfinish(nodeid, location):
    with open({str(log)!r}, "a") as f:
        f.write(nodeid + "\\n")
""")
        pytester.makepyfile(self._parallel_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    def test_logfinish_serial(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logfinish(nodeid, location):
    with open({str(log)!r}, "a") as f:
        f.write(nodeid + "\\n")
""")
        pytester.makepyfile(self._serial_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    # -----------------------------------------------------------------------
    # pytest_runtest_logreport
    # -----------------------------------------------------------------------

    def test_logreport_parallel(self, pytester, tmp_path):
        """Three phases (setup/call/teardown) × 3 items = 9 calls."""
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logreport(report):
    with open({str(log)!r}, "a") as f:
        f.write(report.when + "\\n")
""")
        pytester.makepyfile(self._parallel_src())
        pytester.runpytest().assert_outcomes(passed=3)
        calls = self._read(log)
        assert len(calls) == 9
        assert calls.count("setup") == 3
        assert calls.count("call") == 3
        assert calls.count("teardown") == 3

    def test_logreport_serial(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl
def pytest_runtest_logreport(report):
    with open({str(log)!r}, "a") as f:
        f.write(report.when + "\\n")
""")
        pytester.makepyfile(self._serial_src())
        pytester.runpytest().assert_outcomes(passed=3)
        calls = self._read(log)
        assert len(calls) == 9
        assert calls.count("setup") == 3
        assert calls.count("call") == 3
        assert calls.count("teardown") == 3

    # -----------------------------------------------------------------------
    # pytest_runtest_makereport — required by xfail, flaky, coverage, etc.
    # -----------------------------------------------------------------------

    def test_makereport_parallel(self, pytester, tmp_path):
        """A hookwrapper on makereport must be called for every phase of every item."""
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    with open({str(log)!r}, "a") as f:
        f.write(rep.when + "\\n")
""")
        pytester.makepyfile(self._parallel_src())
        pytester.runpytest().assert_outcomes(passed=3)
        calls = self._read(log)
        assert len(calls) == 9, f"expected 9 makereport calls, got {len(calls)}"
        assert calls.count("setup") == 3
        assert calls.count("call") == 3
        assert calls.count("teardown") == 3

    def test_makereport_serial(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    with open({str(log)!r}, "a") as f:
        f.write(rep.when + "\\n")
""")
        pytester.makepyfile(self._serial_src())
        pytester.runpytest().assert_outcomes(passed=3)
        calls = self._read(log)
        assert len(calls) == 9, f"expected 9 makereport calls, got {len(calls)}"
        assert calls.count("setup") == 3
        assert calls.count("call") == 3
        assert calls.count("teardown") == 3

    # -----------------------------------------------------------------------
    # xfail — requires pytest_runtest_makereport
    # -----------------------------------------------------------------------

    def test_xfail_parallel(self, pytester):
        pytester.makepyfile("""
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
@pytest.mark.xfail
def test_items(n):
    assert False, "expected failure"
""")
        pytester.runpytest().assert_outcomes(xfailed=3)

    def test_xfail_serial(self, pytester):
        pytester.makepyfile("""
import pytest

@pytest.fixture(scope="session")
def sess():
    return 42

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
@pytest.mark.xfail
def test_items(n, sess):
    assert False, "expected failure"
""")
        pytester.runpytest().assert_outcomes(xfailed=3)

    # -----------------------------------------------------------------------
    # pytest_runtest_protocol — hookwrapper variant (both paths)
    # -----------------------------------------------------------------------

    def test_protocol_hookwrapper_serial(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    with open({str(log)!r}, "a") as f:
        f.write(item.nodeid + "\\n")
    yield
""")
        pytester.makepyfile(self._serial_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    # -----------------------------------------------------------------------
    # True serial-setup path (indirect parametrize)
    # -----------------------------------------------------------------------

    def test_makereport_indirect_path(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    rep = outcome.get_result()
    with open({str(log)!r}, "a") as f:
        f.write(rep.when + "\\n")
""")
        pytester.makepyfile(self._indirect_src())
        pytester.runpytest().assert_outcomes(passed=3)
        calls = self._read(log)
        assert len(calls) == 9
        assert calls.count("setup") == 3
        assert calls.count("call") == 3
        assert calls.count("teardown") == 3

    def test_protocol_hookwrapper_indirect_path(self, pytester, tmp_path):
        log = tmp_path / "log.txt"
        pytester.makeconftest(f"""
import pytest

@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    with open({str(log)!r}, "a") as f:
        f.write(item.nodeid + "\\n")
    yield
""")
        pytester.makepyfile(self._indirect_src())
        pytester.runpytest().assert_outcomes(passed=3)
        assert len(self._read(log)) == 3

    def test_xfail_indirect_path(self, pytester):
        pytester.makepyfile("""
import pytest

@pytest.fixture(scope="module")
def mod_fix(request):
    return request.param

@pytest.mark.swarm
@pytest.mark.parametrize("mod_fix", ["a", "b", "c"], indirect=True)
@pytest.mark.xfail
def test_items(mod_fix):
    assert False, "expected failure"
""")
        pytester.runpytest().assert_outcomes(xfailed=3)

    # -----------------------------------------------------------------------
    # pytest.mark.skip
    # -----------------------------------------------------------------------

    def test_skip_parallel(self, pytester):
        pytester.makepyfile("""
import pytest

@pytest.mark.swarm
@pytest.mark.parametrize("n", range(3))
@pytest.mark.skip(reason="just skipping")
def test_items(n):
    pass
""")
        pytester.runpytest().assert_outcomes(skipped=3)

    def test_skip_indirect_path(self, pytester):
        pytester.makepyfile("""
import pytest

@pytest.fixture(scope="module")
def mod_fix(request):
    return request.param

@pytest.mark.swarm
@pytest.mark.parametrize("mod_fix", ["a", "b", "c"], indirect=True)
@pytest.mark.skip(reason="just skipping")
def test_items(mod_fix):
    pass
""")
        pytester.runpytest().assert_outcomes(skipped=3)
