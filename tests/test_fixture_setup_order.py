from __future__ import annotations

from ._helpers import _events


class TestBroadScopeSetupOrder:
    """
    Broad-scope fixtures are pre-fetched in the main thread before worker
    threads start. That prefetch must set them up in the same order real
    pytest would: a fixture's own dependencies are resolved in the order
    they are listed in its signature, right before its own body runs - not
    independently, out of turn, just because they happen to share a scope
    with their dependent.
    """

    def test_sibling_deps_resolved_in_argname_order(self, pytester):
        """fixture1 depends on fixture2, fixture3, fixture4 (declared in that
        order); all four share session scope. Real pytest sets up
        fixture2 -> fixture3 -> fixture4 -> fixture1. Nothing here should let
        a leaf like fixture4 jump ahead of its siblings or of fixture1."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def fixture2():
                open({counter!r}, "a").write("fixture2\\n")
                yield "f2"

            @pytest.fixture(scope="session")
            def fixture3():
                open({counter!r}, "a").write("fixture3\\n")
                yield "f3"

            @pytest.fixture(scope="session")
            def fixture4():
                open({counter!r}, "a").write("fixture4\\n")
                yield "f4"

            @pytest.fixture(scope="session")
            def fixture1(fixture2, fixture3, fixture4):
                open({counter!r}, "a").write("fixture1\\n")
                yield "f1"

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, fixture1):
                assert fixture1 == "f1"
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        assert _events(counter) == ["fixture2", "fixture3", "fixture4", "fixture1"]

    def test_multilevel_chain_resolved_depth_first(self, pytester):
        """test_thing(x, a), a -> b -> c: real pytest sets up x, c, b, a."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def c():
                open({counter!r}, "a").write("c\\n")
                yield "c"

            @pytest.fixture(scope="session")
            def b(c):
                open({counter!r}, "a").write("b\\n")
                yield "b"

            @pytest.fixture(scope="session")
            def a(b):
                open({counter!r}, "a").write("a\\n")
                yield "a"

            @pytest.fixture(scope="session")
            def x():
                open({counter!r}, "a").write("x\\n")
                yield "x"

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, x, a):
                assert (x, a) == ("x", "a")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        assert _events(counter) == ["x", "c", "b", "a"]
