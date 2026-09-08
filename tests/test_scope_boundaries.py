"""Broad-scope fixtures must end where their scope ends, not where the last swarm
group happens to be.

Swarm items bypass pytest's SetupState, so the plugin keeps its own stack of live
broad-scope instances (``BroadScopeCache``). The stack only ever advanced when a
*swarm* item crossed a boundary, so a package whose tests are all marked, followed
by a package whose tests are not, left its package fixture alive until the end of
the session — its teardown ran after the next package had already been set up and
torn down.

The oracle throughout is a plain sequential run of the same tree: whatever pytest
does without the marker is what the plugin must reproduce with it.
"""

from __future__ import annotations

from ._helpers import _events

_SWARM = "@pytest.mark.swarm(max_workers=4)"


# ---------------------------------------------------------------------------
# package scope
# ---------------------------------------------------------------------------

def test_package_fixture_torn_down_before_next_package(pytester):
    """A swarm-only package must finalize before the next package starts.

    The regression case: pkg1 holds nothing but swarm tests, pkg2 nothing but plain
    ones. No swarm item follows pkg1, so nothing used to advance the boundary and
    pkg1's teardown fell through to the end of the session.
    """
    ev = str(pytester.path / "ev.txt")
    pytester.makefile(".py", **{"pkg1/__init__": "", "pkg2/__init__": ""})
    pytester.makefile(".py", **{"pkg1/conftest": f"""
import pytest

@pytest.fixture(scope="package", autouse=True)
def package1():
    open({ev!r}, "a").write("SETUP1\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN1\\n")
"""})
    pytester.makefile(".py", **{"pkg2/conftest": f"""
import pytest

@pytest.fixture(scope="package", autouse=True)
def package2():
    open({ev!r}, "a").write("SETUP2\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN2\\n")
"""})
    pytester.makefile(".py", **{"pkg1/test_1": f"""
import pytest

{_SWARM}
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_1(n):
    assert n
"""})
    pytester.makefile(".py", **{"pkg2/test_2": """
import pytest

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_2(n):
    assert n
"""})

    pytester.runpytest("-v").assert_outcomes(passed=8)
    assert _events(ev) == ["SETUP1", "TEARDOWN1", "SETUP2", "TEARDOWN2"]


# ---------------------------------------------------------------------------
# module scope
# ---------------------------------------------------------------------------

def test_module_fixture_torn_down_before_next_module(pytester):
    """A swarm module's module-scope fixture ends when the module does."""
    ev = str(pytester.path / "ev.txt")
    pytester.makefile(".py", **{"test_a": f"""
import pytest

@pytest.fixture(scope="module", autouse=True)
def mod_a():
    open({ev!r}, "a").write("SETUP_A\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN_A\\n")

{_SWARM}
@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_a(n):
    assert n
"""})
    pytester.makefile(".py", **{"test_b": f"""
import pytest

@pytest.fixture(scope="module", autouse=True)
def mod_b():
    open({ev!r}, "a").write("SETUP_B\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN_B\\n")

@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_b(n):
    assert n
"""})

    pytester.runpytest("-v").assert_outcomes(passed=8)
    assert _events(ev) == ["SETUP_A", "TEARDOWN_A", "SETUP_B", "TEARDOWN_B"]


# ---------------------------------------------------------------------------
# class scope
# ---------------------------------------------------------------------------

def test_class_fixture_torn_down_before_next_class(pytester):
    """A swarm class's class-scope fixture ends when the class does.

    Both classes share one fixture definition, so the expected log is the ordinary
    one instance per class — the swarm class must not keep its instance alive into
    the plain class that follows.
    """
    ev = str(pytester.path / "ev.txt")
    pytester.makepyfile(f"""
import pytest

@pytest.fixture(scope="class", autouse=True)
def cls_res():
    open({ev!r}, "a").write("SETUP\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN\\n")

class TestFirst:
    {_SWARM}
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_swarm(self, n):
        assert n

class TestSecond:
    @pytest.mark.parametrize("n", [1, 2, 3, 4])
    def test_plain(self, n):
        assert n
""")

    pytester.runpytest("-v").assert_outcomes(passed=8)
    assert _events(ev) == ["SETUP", "TEARDOWN", "SETUP", "TEARDOWN"]


# ---------------------------------------------------------------------------
# nested packages — compared against a plain sequential run
# ---------------------------------------------------------------------------

def _make_nested_tree(pytester, root: str, marker: str) -> str:
    """Build ``<root>/`` — a package fixture, a test in it, and one in a subpackage.

    Returns the path of the event log the tree writes to.
    """
    ev = str(pytester.path / f"{root}.txt")
    pytester.makefile(".py", **{f"{root}/__init__": "", f"{root}/sub/__init__": ""})
    pytester.makefile(".py", **{f"{root}/conftest": f"""
import pytest

@pytest.fixture(scope="package", autouse=True)
def pkg_res():
    open({ev!r}, "a").write("SETUP\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN\\n")
"""})
    pytester.makefile(".py", **{f"{root}/test_top": f"""
import pytest

{marker}
@pytest.mark.parametrize("n", [1, 2, 3])
def test_top(n):
    assert n
"""})
    pytester.makefile(".py", **{f"{root}/sub/test_deep": f"""
import pytest

{marker}
@pytest.mark.parametrize("n", [1, 2, 3])
def test_deep(n):
    assert n
"""})
    return ev


def test_nested_package_fixture_matches_sequential(pytester):
    """Descending into a subpackage must not disturb the outer package fixture.

    Scope boundaries used to be compared by directory path, so ``pkg/sub`` read as a
    different package and tore ``pkg`` down on the way in. Node identity is what
    fixes it: an item in ``pkg/sub`` still has ``Package(pkg)`` in its chain.

    The expected log is whatever the same tree produces with the marker removed.
    """
    swarm_ev = _make_nested_tree(pytester, "swarmed", _SWARM)
    plain_ev = _make_nested_tree(pytester, "plain", "")

    pytester.runpytest("swarmed", "-v").assert_outcomes(passed=6)
    pytester.runpytest("plain", "-v").assert_outcomes(passed=6)

    assert _events(swarm_ev) == _events(plain_ev)


# ---------------------------------------------------------------------------
# session scope survives everything
# ---------------------------------------------------------------------------

def test_session_fixture_survives_package_boundaries(pytester):
    """Advancing the boundary on every item must not shorten session scope.

    Only the swarm test requests the fixture, so exactly one instance exists; the
    plain tests in the next package must neither finalize it nor make its teardown
    run before the session ends.
    """
    ev = str(pytester.path / "ev.txt")
    pytester.makefile(".py", **{"pkg1/__init__": "", "pkg2/__init__": ""})
    pytester.makefile(".py", **{"conftest": f"""
import pytest

@pytest.fixture(scope="session")
def sess():
    open({ev!r}, "a").write("SETUP\\n")
    yield
    open({ev!r}, "a").write("TEARDOWN\\n")
"""})
    pytester.makefile(".py", **{"pkg1/test_1": f"""
import pytest

{_SWARM}
@pytest.mark.parametrize("n", [1, 2, 3])
def test_1(n, sess):
    assert n
"""})
    pytester.makefile(".py", **{"pkg2/test_2": f"""
import pytest

@pytest.mark.parametrize("n", [1, 2, 3])
def test_2(n):
    open({ev!r}, "a").write("PLAIN\\n")
"""})

    pytester.runpytest("-v").assert_outcomes(passed=6)
    assert _events(ev) == ["SETUP", "PLAIN", "PLAIN", "PLAIN", "TEARDOWN"]
