from __future__ import annotations

import pytest

from ._helpers import _events


class TestTransitiveDeps:
    """
    Scenarios where a broad-scope (session/module) fixture is a transitive
    dependency of a function-scope fixture with params.
    The broad-scope fixture must be initialised exactly once regardless of how
    many threads invoke the function-scope fixture in parallel.
    """

    def test_session_dep_via_function_params(self, pytester):
        """
        session-scope fixture as a dependency of a function-scope fixture with params.
        Pattern: session_fixture → function_fixture(params=[...]) → test.
        SETUP count for session_fixture must be 1 regardless of the number of parallel threads.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def session_res():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 42}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.fixture(params=[1, 2, 3, 4, 5])
            def fn_fixture(session_res, request):
                return (request.param, session_res["val"])

            @pytest.mark.swarm(max_workers=5)
            def test_thing(fn_fixture):
                param, val = fn_fixture
                assert val == 42
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=5)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_module_dep_via_function_params(self, pytester):
        """
        module-scope fixture as a dependency of a function-scope fixture with params.
        SETUP count for module_fixture must be 1.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def module_res():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 7}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.fixture(params=["a", "b", "c", "d"])
            def fn_fixture(module_res, request):
                return (request.param, module_res["val"])

            @pytest.mark.swarm(max_workers=4)
            def test_thing(fn_fixture):
                param, val = fn_fixture
                assert val == 7
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_multilevel_session_dep(self, pytester):
        """
        Multi-level chain: session_res → mid_fixture (function) → leaf_fixture (function, params) → test.
        session_res is created exactly once; mid and leaf are per-item.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def session_res():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"id": 123}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.fixture
            def mid_fixture(session_res):
                return session_res["id"]

            @pytest.fixture(params=range(6))
            def leaf_fixture(mid_fixture, request):
                return (request.param, mid_fixture)

            @pytest.mark.swarm(max_workers=6)
            def test_thing(leaf_fixture):
                param, session_id = leaf_fixture
                assert session_id == 123
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=6)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_session_dep_value_identical_across_threads(self, pytester):
        """
        All parallel threads receive the same session-fixture instance (not a copy):
        verified by comparing id() of the object.
        """
        ids_file = str(pytester.path / "ids.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def session_obj():
                return object()

            @pytest.fixture(params=range(8))
            def fn_fixture(session_obj, request):
                return id(session_obj)

            @pytest.mark.swarm(max_workers=8)
            def test_thing(fn_fixture):
                open({ids_file!r}, "a").write(str(fn_fixture) + "\\n")
                assert True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=8)
        ids = _events(ids_file)
        assert len(ids) == 8
        assert len(set(ids)) == 1, f"Expected a single object id, got: {set(ids)}"
