# pytest-swarm

A pytest plugin that runs parametrized test variants in parallel threads — with correct fixture lifecycle.

## What problem does it solve?

When a test is parametrized over many values (hosts, configs, datasets), pytest
runs every variant **sequentially** by default. If each variant takes 2 s and you
have 50 variants, the suite takes 100 s even though the variants are completely
independent.

`pytest-swarm` solves exactly this one problem: it runs the variants of a single
`@pytest.mark.swarm`-decorated test in parallel threads, cutting wall-clock time
to roughly `max(variant_times)` instead of `sum(variant_times)`.

### Not a replacement for pytest-xdist or pytest-parallel

`pytest-xdist` distributes **entire tests** across processes or machines.
`pytest-parallel` also parallelizes at the test level.

`pytest-swarm` operates one level lower: it parallelizes the **variants of a
single parametrized test** while everything else — fixture lifecycle, test
ordering, reporting — stays exactly as in a normal sequential run. The two
approaches complement each other; you can use `pytest-swarm` alongside
`pytest-xdist`.

### Thread safety is your responsibility

Worker threads share the same process. Any **shared mutable state** accessed from
parallel test bodies — global variables, module-level caches, resources held in
broad-scope fixtures — must be protected by locks or other synchronization
primitives. The plugin guarantees that broad-scope fixtures are created once in
the main thread, but it does not add any locking around how you *use* them inside
the test body.

## Installation

```bash
pip install pytest-swarm
```

## Quick start

```python
import pytest
import time

@pytest.fixture(scope="module")
def config():
    return load_config()  # created once, shared by all variants

@pytest.fixture
def client(config):
    return Client(config)  # created in parallel, one per variant

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("host", ["h1", "h2", "h3", "h4"])
def test_ping(host, client):
    time.sleep(1)  # 4 variants run simultaneously → ~1 s total, not ~4 s
```

Without `max_workers` — uses CPU count.

## Controlling worker count

**Priority chain (highest to lowest):**

1. `@pytest.mark.swarm(max_workers=N)` — per-test marker
2. `--swarm-workers=N` — CLI option
3. `PYTEST_SWARM_WORKERS=N` — environment variable
4. `os.cpu_count()` — default

### CLI

```bash
pytest --swarm-workers=4
```

### Environment variable

```bash
PYTEST_SWARM_WORKERS=4 pytest
```

Useful for CI pipelines where you want a consistent cap without modifying test
code. The CLI option takes priority over the env variable.

### Per-test override

```python
@pytest.mark.swarm(max_workers=16)  # overrides both CLI and env var
@pytest.mark.parametrize("host", hosts)
def test_connect(host):
    ...
```

## Fixture scope behavior

The plugin respects pytest fixture scopes. Behavior depends on whether fixtures
are function-scoped or broader.

### Function-scope — full parallel lifecycle

Each parametrized variant runs its **entire lifecycle** (setup → call → teardown)
in its own thread. Fixture setup runs in parallel — useful when the fixture itself
is expensive (e.g. establishing a connection).

```python
@pytest.fixture
def connection(request):
    conn = connect(request.param)  # runs in parallel across threads
    yield conn
    conn.close()

@pytest.mark.swarm(max_workers=6)
@pytest.mark.parametrize("connection", hosts, indirect=True)
def test_command(connection):
    assert connection.run("uptime")
```

Works with all function-scope parametrization forms:

| Form | Works |
|---|---|
| `@pytest.mark.parametrize("n", [...])` | ✓ |
| `@pytest.fixture(params=[...])` | ✓ |
| `@pytest.mark.parametrize("fix", [...], indirect=True)` | ✓ |
| fixture depending on another fixture | ✓ |
| `pytestmark = pytest.mark.usefixtures(...)` | ✓ |

### Class / module / package / session scope — shared instance, parallel calls

Broad-scope fixtures are created **once** in the main thread and shared between
all parallel test variants. Only the test body runs in parallel.

```python
@pytest.fixture(scope="module")
def pool():
    return ConnectionPool(size=10)  # created once; must be thread-safe

@pytest.mark.swarm(max_workers=10)
@pytest.mark.parametrize("cmd", commands)
def test_run(cmd, pool):
    pool.run(cmd)  # parallel — pool must handle concurrent access
```

SETUP and TEARDOWN happen exactly as many times as they would in a normal
sequential run — once per fixture scope boundary.

### Mixed scopes

When a test uses both function-scope and broad-scope fixtures, the broad-scope
fixture is created once (serial), and each variant gets its own function-scope
instance (parallel setup).

## Autouse fixtures from plugins

Autouse session/module-scope fixtures from third-party plugins (e.g.
`_session_faker` from pytest-Faker) are present in `item.fixturenames` but are
correctly ignored when deciding the execution path. They don't force a serial
fallback.

## Non-parallel tests

Tests without `@pytest.mark.swarm` run normally — the plugin does not affect
them. Parallel and sequential tests can coexist freely in the same session.

```python
def test_sequential():  # unaffected
    ...

@pytest.mark.swarm(max_workers=4)
@pytest.mark.parametrize("n", range(4))
def test_parallel(n):  # parallel
    ...
```

## Plugin compatibility

Most pytest hooks work correctly for swarm tests:

| Hook | Works |
|---|---|
| `pytest_runtest_logstart` / `logfinish` / `logreport` | ✓ |
| `pytest_runtest_makereport` (hookwrapper) | ✓ |
| `pytest_runtest_protocol` (hookwrapper) | ✓ |
| `pytest_runtest_protocol` (regular `@pytest.hookimpl`) | ✗ see below |
| `pytest.mark.xfail` / `pytest.mark.skip` | ✓ |

### `pytest_runtest_protocol` — hookwrapper only

`pytest_runtest_protocol` is a *firstresult* hook: the first implementation that
returns a non-`None` value wins and the rest of the chain is skipped. The plugin
implements this hook with `tryfirst=True` and returns `True` to prevent pytest's
default runner from executing the test again via `SetupState`.

Because of this, a regular implementation in a conftest —

```python
# conftest.py
@pytest.hookimpl  # does NOT fire for swarm tests
def pytest_runtest_protocol(item, nextitem):
    ...
```

— is never reached: the plugin's `tryfirst` implementation runs first and stops
the chain. This does **not** affect hookwrappers, which always wrap the full chain
regardless of `firstresult`:

```python
@pytest.hookimpl(hookwrapper=True)  # works correctly
def pytest_runtest_protocol(item, nextitem):
    ...
    yield
```

In practice this is rarely a concern: the overwhelming majority of plugins that
observe `pytest_runtest_protocol` (pytest-xdist, pytest-rerunfailures, etc.) use
`hookwrapper=True`.

## Limitations

- **Broad-scope fixture setup is serial.** Only the test body is parallelized when
  class/module/package/session fixtures are involved. Move expensive operations into the
  test body or use a connection pool pattern to work around this.

- **Thread safety is the test's responsibility.** Shared mutable state accessed
  from parallel test bodies must be protected by locks or other synchronization.

- **Built-in pytest fixtures** (`tmp_path`, `capfd`, `monkeypatch`, …) in the
  fixture dependency chain may not work in the parallel-setup path. The plugin
  falls back to serial setup automatically when it detects them.

## How it works

```
function-scope group                         time →

thread 1  [ setup ][ test body ][ teardown ]
thread 2  [ setup ][ test body ][ teardown ]  ← all run simultaneously
thread 3  [ setup ][ test body ][ teardown ]


broad-scope group                            time →

main      [ setup ]                [ teardown ]
thread 1           [ test body ]
thread 2           [ test body ]              ← all run simultaneously
thread 3           [ test body ]
```

Fixture functions are called directly inside threads (bypassing pytest's
`SetupState`), so function-scope setup runs truly in parallel. Broad-scope
fixtures go through normal pytest setup in the main thread to preserve shared
instance semantics.
