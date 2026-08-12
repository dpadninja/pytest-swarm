from __future__ import annotations


class TestBroadScopeFixtureErrors:
    """
    Broad-scope (session/package/module/class) fixtures are pre-fetched in the
    main thread, ahead of and outside any per-item CallInfo wrapper. A
    non-AssertionError exception raised there must still be reported as a
    normal per-test setup error - not escape pytest_runtestloop and crash the
    whole run with INTERNALERROR.
    """

    def test_session_scope_fixture_raises(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="session")
            def broken():
                raise ValueError("boom")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, broken):
                assert True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(errors=3)
        assert "INTERNALERROR" not in result.stdout.str()
        result.stdout.fnmatch_lines(["*ValueError*boom*"])

    def test_module_scope_transitive_dependency_raises(self, pytester):
        """The failure is in a dependency of the fixture the test asks for,
        reached only transitively during prefetch."""
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="module")
            def dep():
                raise RuntimeError("dep boom")

            @pytest.fixture(scope="module")
            def broken(dep):
                return dep

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2, 3])
            def test_thing(n, broken):
                assert True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(errors=3)
        assert "INTERNALERROR" not in result.stdout.str()
        result.stdout.fnmatch_lines(["*RuntimeError*dep boom*"])

    def test_other_group_unaffected_by_prior_group_prefetch_error(self, pytester):
        """A prefetch failure in one swarm group must not affect a later,
        unrelated swarm group."""
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="module")
            def broken():
                raise ValueError("boom")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2])
            def test_fails(n, broken):
                assert True

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2])
            def test_ok(n):
                assert True
        """)
        result = pytester.runpytest("-v")
        result.assert_outcomes(errors=2, passed=2)
        assert "INTERNALERROR" not in result.stdout.str()
