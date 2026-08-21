"""Fixtures must still be alive while the test bodies they belong to are running.

Regression tests for a path that split a group into phases - all setups, then all
bodies in parallel, then teardowns. Two independent mechanisms finalized fixtures
during the setup phase, before any body had run:

* mechanism A - ``SetupState.teardown_exact`` called between setups sweeps the
  stack down to the common ancestor, running function-scoped finalizers;
* mechanism B - ``FixtureDef`` holds a single live value, so setting up an
  indirect-parametrized broad-scope fixture for the next item tears down the
  previous item's instance.

Both are now avoided by resolving fixtures per item in worker threads, and per
parameter value in the main thread, instead of driving pytest's own machinery.
"""

from __future__ import annotations

import pytest

from ._helpers import _events


def _nothing_torn_down_before_first_body(events: list[str]) -> None:
    assert "BODY" in events, events
    first_body = events.index("BODY")
    assert "TEARDOWN" not in events[:first_body], events


class TestMechanismA:
    """Function-scoped fixtures must not be finalized before their body runs."""

    def test_function_fixture_alive_during_body(self, pytester):
        ev = str(pytester.path / "ev.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture
            def res():
                open({ev!r}, "a").write("SETUP\\n")
                yield "r"
                open({ev!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("i", range(4))
            def test_thing(res, tmp_path, i):
                open({ev!r}, "a").write("BODY\\n")
        """)
        pytester.runpytest().assert_outcomes(passed=4)
        _nothing_torn_down_before_first_body(_events(ev))

    def test_closed_resource_is_not_handed_to_the_body(self, pytester):
        """The visible symptom: the body gets an already-finalized object."""
        pytester.makepyfile("""
            import pytest

            @pytest.fixture
            def logfile(tmp_path):
                f = open(tmp_path / "log.txt", "w")
                yield f
                f.close()

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("i", range(4))
            def test_write(logfile, i):
                logfile.write("row %d\\n" % i)
        """)
        pytester.runpytest().assert_outcomes(passed=4)

    def test_monkeypatch_in_fixture_stays_applied(self, pytester):
        """Silent variant: the patch is undone before the body observes it."""
        pytester.makepyfile("""
            import os
            import pytest

            @pytest.fixture
            def env(monkeypatch, i):
                monkeypatch.setenv("SWARM_PROBE", "v%d" % i)

            @pytest.mark.swarm(max_workers=1)
            @pytest.mark.parametrize("i", range(2))
            def test_env(env, i):
                assert os.environ["SWARM_PROBE"] == "v%d" % i
        """)
        pytester.runpytest().assert_outcomes(passed=2)


class TestMechanismB:
    """Indirect-parametrized broad-scope fixtures must all stay live at once."""

    def test_indirect_module_fixture_alive_during_body(self, pytester):
        ev = str(pytester.path / "ev.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                open({ev!r}, "a").write("SETUP\\n")
                yield "conn-%s" % request.param
                open({ev!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("conn", [1, 2, 3], indirect=True)
            def test_thing(conn):
                open({ev!r}, "a").write("BODY\\n")
        """)
        pytester.runpytest().assert_outcomes(passed=3)
        _nothing_torn_down_before_first_body(_events(ev))

    def test_indirect_value_matches_the_item(self, pytester):
        """Each item must see its own instance, not the last one set up."""
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                obj = {"val": request.param, "closed": False}
                yield obj
                obj["closed"] = True

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("conn", [1, 2, 3], indirect=True)
            def test_thing(conn):
                assert conn["closed"] is False
        """)
        pytester.runpytest().assert_outcomes(passed=3)


class TestSingleItemGroup:
    """A one-item group is the degenerate case - setup, body, teardown, in order."""

    def test_single_item_with_builtin_fixture(self, pytester):
        ev = str(pytester.path / "ev.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture
            def res():
                yield "r"
                open({ev!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm
            def test_thing(res, tmp_path):
                open({ev!r}, "a").write("BODY\\n")
        """)
        pytester.runpytest().assert_outcomes(passed=1)
        assert _events(ev) == ["BODY", "TEARDOWN"]
