"""Cases that used to fall off the parallel path and now stay on it.

Built-in fixtures the plugin can resolve itself, and broad-scope fixtures that are
parametrized, both used to demote a group to a half-parallel path that finalized
fixtures before the test bodies ran. They now run fully parallel, one fixture
instance per item (or per parameter value), with teardown at the right boundary.
"""

from __future__ import annotations

import pytest

from ._helpers import _events


class TestBuiltinFixtures:

    def test_tmp_path_stays_on_the_parallel_path(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.mark.swarm(max_workers=8)
            @pytest.mark.parametrize("i", range(8))
            def test_thing(tmp_path, i):
                (tmp_path / "f.txt").write_text(str(i))
                assert (tmp_path / "f.txt").read_text() == str(i)
        """)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=8)
        result.stdout.fnmatch_lines(["parallel*8 item(s)*"])

    def test_each_item_gets_its_own_tmp_path(self, pytester):
        paths = str(pytester.path / "paths.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("i", range(4))
            def test_thing(tmp_path, i):
                open({paths!r}, "a").write(str(tmp_path) + "\\n")
        """)
        pytester.runpytest().assert_outcomes(passed=4)
        recorded = _events(paths)
        assert len(recorded) == 4
        assert len(set(recorded)) == 4, recorded

    def test_unsupported_builtin_falls_back_to_sequential(self, pytester):
        """capsys redirects process-global file descriptors — threads cannot share it."""
        pytester.makepyfile("""
            import pytest

            @pytest.mark.swarm
            @pytest.mark.parametrize("i", range(3))
            def test_thing(capsys, i):
                print("hello %d" % i)
                assert capsys.readouterr().out == "hello %d\\n" % i
        """)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["sequential*3 item(s)*"])

    def test_user_fixture_shadowing_a_builtin_name_is_not_a_blocker(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.fixture
            def cache():
                return {"mine": True}

            @pytest.mark.swarm
            @pytest.mark.parametrize("i", range(3))
            def test_thing(cache, i):
                assert cache["mine"]
        """)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=3)
        result.stdout.fnmatch_lines(["parallel*3 item(s)*"])


class TestParametrizedBroadScope:

    def test_indirect_module_fixture_stays_on_the_parallel_path(self, pytester):
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                yield "conn-%s" % request.param

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("conn", [1, 2, 3, 4], indirect=True)
            def test_thing(conn):
                assert conn.startswith("conn-")
        """)
        result = pytester.runpytest("--swarm-explain")
        result.assert_outcomes(passed=4)
        result.stdout.fnmatch_lines(["parallel*4 item(s)*"])

    def test_one_instance_per_parameter_value(self, pytester):
        counter = str(pytester.path / "counter.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                open({counter!r}, "a").write("SETUP\\n")
                yield request.param
                open({counter!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("n", [1, 2])
            @pytest.mark.parametrize("conn", ["a", "b"], indirect=True)
            def test_thing(conn, n):
                assert conn in ("a", "b")
        """)
        pytester.runpytest().assert_outcomes(passed=4)
        ev = _events(counter)
        assert ev.count("SETUP") == 2, ev
        assert ev.count("TEARDOWN") == 2, ev

    def test_dependent_fixture_follows_the_parameter(self, pytester):
        """A broad-scope fixture that depends on a parametrized one is parametrized too."""
        pytester.makepyfile("""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                return "conn-%s" % request.param

            @pytest.fixture(scope="module")
            def pool(conn):
                return "pool-of-" + conn

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("conn", [1, 2, 3], indirect=True)
            def test_thing(conn, pool):
                assert pool == "pool-of-" + conn
        """)
        pytester.runpytest().assert_outcomes(passed=3)

    def test_instances_stay_alive_until_the_scope_ends(self, pytester):
        ev = str(pytester.path / "ev.txt")
        pytester.makepyfile(f"""
            import pytest

            @pytest.fixture(scope="module")
            def conn(request):
                yield "conn-%s" % request.param
                open({ev!r}, "a").write("TEARDOWN\\n")

            @pytest.mark.swarm(max_workers=4)
            @pytest.mark.parametrize("conn", [1, 2, 3], indirect=True)
            def test_thing(conn):
                open({ev!r}, "a").write("BODY\\n")
        """)
        pytester.runpytest().assert_outcomes(passed=3)
        ev_list = _events(ev)
        assert ev_list[:3] == ["BODY", "BODY", "BODY"], ev_list
        assert ev_list.count("TEARDOWN") == 3, ev_list
