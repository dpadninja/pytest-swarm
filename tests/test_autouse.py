from __future__ import annotations

import time

import pytest

from ._helpers import _events


# ---------------------------------------------------------------------------
# Autouse broad-scope fixture from a plugin
# ---------------------------------------------------------------------------

def test_autouse_session_fixture_does_not_block_swarm_setup(pytester):
    """
    A session-scope autouse fixture is present in item.fixturenames but must
    not push the group onto the serial-setup path.
    """
    pytester.makeconftest("""
import pytest

@pytest.fixture(scope="session", autouse=True)
def session_autouse():
    pass
""")
    pytester.makepyfile(f"""
        import pytest, time

        @pytest.fixture
        def slow_fn_fixture(request):
            time.sleep(0.4)
            yield request.param

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("slow_fn_fixture", [1, 2, 3, 4], indirect=True)
        def test_thing(slow_fn_fixture):
            assert True
    """)
    t0 = time.monotonic()
    result = pytester.runpytest("-v")
    elapsed = time.monotonic() - t0
    result.assert_outcomes(passed=4)
    assert elapsed < 1.2, (
        f"Autouse session fixture blocked parallel setup: elapsed={elapsed:.2f}s"
    )


def test_autouse_function_fixture_runs_per_item(pytester):
    """
    A function-scope autouse fixture must run for every swarm item (setup +
    teardown), even when it is not listed explicitly in the test arguments.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makeconftest(f"""
import pytest

@pytest.fixture(autouse=True)
def fn_autouse():
    open({counter!r}, "a").write("SETUP\\n")
    yield
    open({counter!r}, "a").write("TEARDOWN\\n")
""")
    pytester.makepyfile("""
        import pytest

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_thing(n):
            pass
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert ev.count("SETUP") == 4
    assert ev.count("TEARDOWN") == 4


def test_autouse_session_fixture_runs_once(pytester):
    """
    A session-scope autouse fixture must run exactly once and must not be
    re-created between swarm groups.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makeconftest(f"""
import pytest

@pytest.fixture(scope="session", autouse=True)
def sess_autouse():
    open({counter!r}, "a").write("SETUP\\n")
    yield
    open({counter!r}, "a").write("TEARDOWN\\n")
""")
    pytester.makepyfile("""
        import pytest

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_first(n):
            pass

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_second(n):
            pass
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=8)
    ev = _events(counter)
    assert ev.count("SETUP") == 1
    assert ev.count("TEARDOWN") == 1


def test_autouse_package_fixture_runs_once_across_modules(pytester):
    """
    A package-scope autouse fixture must be created once for all swarm tests
    across different modules in the same package.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makefile(".py", **{"pkg/__init__": ""})
    pytester.makefile(".py", **{"pkg/conftest": f"""
import pytest

@pytest.fixture(scope="package", autouse=True)
def pkg_autouse():
    open({counter!r}, "a").write("SETUP\\n")
    yield
    open({counter!r}, "a").write("TEARDOWN\\n")
"""})
    pytester.makefile(".py", **{"pkg/test_a": """
import pytest

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_thing(n):
    pass
"""})
    pytester.makefile(".py", **{"pkg/test_b": """
import pytest

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_thing(n):
    pass
"""})
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=8)
    ev = _events(counter)
    assert ev.count("SETUP") == 1
    assert ev.count("TEARDOWN") == 1


# ---------------------------------------------------------------------------
# usefixtures
# ---------------------------------------------------------------------------

def test_usefixtures_marker_runs_fixtures(pytester):
    """
    pytestmark = pytest.mark.usefixtures(...) — fixtures listed in the marker
    must be executed in parallel alongside the test arguments.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makepyfile(f"""
        import pytest

        @pytest.fixture
        def side_effect():
            open({counter!r}, "a").write("SETUP\\n")
            yield
            open({counter!r}, "a").write("TEARDOWN\\n")

        pytestmark = pytest.mark.usefixtures("side_effect")

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_thing(n):
            pass
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert ev.count("SETUP") == 4
    assert ev.count("TEARDOWN") == 4


def test_usefixtures_broad_scope_setup_once(pytester):
    """
    usefixtures with a module-scope fixture: the fixture is initialised in the
    main thread exactly once.
    """
    counter = str(pytester.path / "counter.txt")
    pytester.makepyfile(f"""
        import pytest

        @pytest.fixture(scope="module")
        def shared():
            open({counter!r}, "a").write("SETUP\\n")
            yield {{"ok": True}}
            open({counter!r}, "a").write("TEARDOWN\\n")

        pytestmark = pytest.mark.usefixtures("shared")

        @pytest.mark.swarm(max_workers=4)
        @pytest.mark.parametrize("n", [1, 2, 3, 4])
        def test_thing(n):
            pass
    """)
    result = pytester.runpytest("-v")
    result.assert_outcomes(passed=4)
    ev = _events(counter)
    assert ev.count("SETUP") == 1
    assert ev.count("TEARDOWN") == 1
