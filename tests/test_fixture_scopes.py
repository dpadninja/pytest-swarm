from __future__ import annotations

import pytest

from ._helpers import _events


# ---------------------------------------------------------------------------
# module scope
# ---------------------------------------------------------------------------

class TestModuleScope:

    def test_direct_parametrize_setup_once(self, pytester):
        """@parametrize on the test: fixture is set up and torn down exactly once."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 42}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3, 4])
            def test_thing(n, resource):
                assert resource["val"] == 42
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_fixture_params_setup_per_param_value(self, pytester):
        """Fixture with params=[...] + module scope: exactly 1 SETUP and 1 TEARDOWN per param value."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module", params=["a", "b", "c"])
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            def test_thing(resource):
                assert resource["val"] in ("a", "b", "c")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        ev = _events(counter)
        assert ev.count("SETUP") == 3
        assert ev.count("TEARDOWN") == 3

    def test_indirect_parametrize_setup_per_param_value(self, pytester):
        """indirect=True + module scope: each indirect value gets its own fixture instance."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param * 10}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("resource", [1, 2, 3], indirect=True)
            def test_thing(resource):
                assert resource["val"] in (10, 20, 30)
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        ev = _events(counter)
        assert ev.count("SETUP") == 3
        assert ev.count("TEARDOWN") == 3

    def test_fixture_params_combined_with_test_parametrize(self, pytester):
        """
        Fixture with params + @parametrize on the test simultaneously.
        Total variants = len(fixture_params) × len(test_params).
        SETUP/TEARDOWN count = len(fixture_params): one fixture instance serves
        all test_params for a given fixture_param value.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module", params=["x", "y"])
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, resource):
                assert resource["val"] in ("x", "y")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=6)
        ev = _events(counter)
        assert ev.count("SETUP") == 2
        assert ev.count("TEARDOWN") == 2

    def test_module_fixture_shared_with_function_fixture(self, pytester):
        """
        Mix of module- and function-scope fixtures.
        Module fixture is created once and shared across all threads.
        Function fixture is created per-item in the parallel setup phase.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def mod_res():
                open({counter!r}, "a").write("MOD_SETUP\\n")
                yield {{"shared": True}}
                open({counter!r}, "a").write("MOD_TEARDOWN\\n")

            @pytest.fixture
            def fn_res():
                open({counter!r}, "a").write("FN_SETUP\\n")
                yield {{"own": True}}
                open({counter!r}, "a").write("FN_TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, mod_res, fn_res):
                assert mod_res["shared"] is True
                assert fn_res["own"] is True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        ev = _events(counter)
        assert ev.count("MOD_SETUP") == 1
        assert ev.count("MOD_TEARDOWN") == 1
        assert ev.count("FN_SETUP") == 3
        assert ev.count("FN_TEARDOWN") == 3

    def test_module_fixture_shared_across_swarm_functions(self, pytester):
        """
        module-scope fixture used by two different @swarm functions in the same module:
        must be set up and torn down exactly once.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def shared():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 7}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=6)
            @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
            def test_first(n, shared):
                assert shared["val"] == 7

            @pytest.mark.swarm(max_workers=6)
            @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6])
            def test_second(n, shared):
                assert shared["val"] == 7
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=12)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1


# ---------------------------------------------------------------------------
# session scope
# ---------------------------------------------------------------------------

class TestSessionScope:

    def test_direct_parametrize_setup_once(self, pytester):
        """session scope: 1 SETUP and 1 TEARDOWN for the entire session."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 99}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3, 4])
            def test_thing(n, resource):
                assert resource["val"] == 99
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_session_fixture_shared_across_serial_setup_groups(self, pytester):
        """
        session-scope fixture must not be re-created between groups that take the
        serial-setup path (indirect module-scope + class).
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def sess():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 1}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.fixture(scope="module")
            def mod(request):
                return request.param

            @pytest.fixture
            def fn(request):
                return request.param

            @pytest.mark.swarm
            @pytest.mark.parametrize("fn,mod", [("a", 1), ("b", 2)],
                                     indirect=True, scope="module")
            class TestGroup:
                def test_1(self, fn, mod, sess):
                    assert sess["val"] == 1

                def test_2(self, fn, mod, sess):
                    assert sess["val"] == 1
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_indirect_parametrize(self, pytester):
        """indirect + session scope: 1 SETUP/TEARDOWN per indirect value."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("resource", ["p", "q"], indirect=True)
            def test_thing(resource):
                assert resource["val"] in ("p", "q")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=2)
        ev = _events(counter)
        assert ev.count("SETUP") == 2
        assert ev.count("TEARDOWN") == 2

    def test_transitively_shared_session_fixture_setup_once(self, pytester):
        """
        A session fixture reached only *transitively* - as another broad-scope
        fixture's dependency, not directly by the test - must still be
        pre-fetched and cached exactly once, regardless of the (undefined,
        hash-randomization-dependent) order in which same-scope siblings are
        visited.
        """
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="session")
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                return object()

            @pytest.fixture(scope="session", autouse=True)
            def derived(resource):
                return {{"resource": resource}}

            @pytest.fixture
            def consumer(resource):
                return {{"resource": resource}}

            @pytest.mark.swarm(max_workers=16)
            @pytest.mark.parametrize("n", list(range(16)))
            def test_thing(n, consumer):
                assert consumer["resource"] is not None
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=16)
        ev = _events(counter)
        assert ev.count("SETUP") == 1


# ---------------------------------------------------------------------------
# function scope (default)
# ---------------------------------------------------------------------------

class TestFunctionScope:

    def test_each_variant_gets_own_fixture(self, pytester):
        """function scope: each test variant gets its own fixture instance."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 7}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3, 4])
            def test_thing(n, resource):
                assert resource["val"] == 7
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 4
        assert ev.count("TEARDOWN") == 4

    def test_fixture_params_function_scope(self, pytester):
        """Fixture with params + function scope: each (test_param, fixture_param) combination is independent."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(params=["a", "b"])
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2])
            def test_thing(n, resource):
                assert resource["val"] in ("a", "b")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 4
        assert ev.count("TEARDOWN") == 4

    def test_indirect_parametrize(self, pytester):
        """indirect + function scope: each variant gets its own fixture instance."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param * 10}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("resource", [1, 2, 3], indirect=True)
            def test_thing(resource):
                assert resource["val"] in (10, 20, 30)
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        ev = _events(counter)
        assert ev.count("SETUP") == 3
        assert ev.count("TEARDOWN") == 3


# ---------------------------------------------------------------------------
# class scope
# ---------------------------------------------------------------------------

class TestClassScope:

    def test_direct_parametrize_setup_once_per_class(self, pytester):
        """class scope: 1 SETUP and 1 TEARDOWN per class."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="class")
            def resource():
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": 55}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            class TestGroup:
                @pytest.mark.swarm(max_workers=4)
                @pytest.mark.parametrize("n", [1, 2, 3])
                def test_thing(self, n, resource):
                    assert resource["val"] == 55
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=3)
        ev = _events(counter)
        assert ev.count("SETUP") == 1
        assert ev.count("TEARDOWN") == 1

    def test_fixture_params_class_scope(self, pytester):
        """Fixture with params + class scope: 1 instance per (class, fixture_param) combination."""
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="class", params=["p", "q"])
            def resource(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield {{"val": request.param}}
                open({counter!r}, "a").write("TEARDOWN\\n")

            class TestGroup:
                @pytest.mark.swarm(max_workers=4)
                @pytest.mark.parametrize("n", [1, 2])
                def test_thing(self, n, resource):
                    assert resource["val"] in ("p", "q")
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 2
        assert ev.count("TEARDOWN") == 2
