from __future__ import annotations

from ._helpers import _events


class TestFixtureOverride:
    """
    Standard pytest pattern: a fixture is redefined at a closer scope
    (module/nested conftest) and requests the wider fixture of the *same
    name* as its own dependency, e.g.:

        # conftest.py
        @pytest.fixture
        def resource():
            return "base"

        # test_module.py
        @pytest.fixture
        def resource(resource):
            return "override+" + resource

    The resolver must pick the definition one level up the override chain
    for the self-named dependency, not the overriding fixture itself again
    (which would recurse forever).
    """

    def test_function_scope_override(self, pytester):
        pytester.makeconftest("""
            import pytest

            @pytest.fixture
            def resource():
                return "base"
        """)
        pytester.makepyfile("""
            import pytest

            @pytest.fixture
            def resource(resource):
                return "override+" + resource

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3, 4])
            def test_thing(n, resource):
                assert resource == "override+base"
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)

    def test_session_scope_override(self, pytester):
        """Same pattern, but both the base and the override are session-scoped
        - exercised through the broad-scope prefetch path, not the per-item one."""
        counter = str(pytester.path / "counter.txt")
        pytester.makeconftest(f"""
            import pytest

            @pytest.fixture(scope="session")
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                yield "base"
                open({counter!r}, "a").write("TEARDOWN\\n")
        """)
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="session")
            def resource(resource):
                return "override+" + resource

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3, 4])
            def test_thing(n, resource):
                assert resource == "override+base"
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_multilevel_override_chain(self, pytester):
        """conftest -> nested conftest -> test module, each overriding the same name."""
        pytester.makeconftest("""
            import pytest

            @pytest.fixture
            def resource():
                return "base"
        """)
        sub = pytester.mkpydir("sub")
        (sub / "conftest.py").write_text("""
import pytest

@pytest.fixture
def resource(resource):
    return "mid+" + resource
""")
        (sub / "test_thing.py").write_text("""
import pytest

@pytest.fixture
def resource(resource):
    return "top+" + resource

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", [1, 2])
def test_thing(n, resource):
    assert resource == "top+mid+base"
""")
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)
