from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pytest-swarm")
except PackageNotFoundError:
    __version__ = "unknown"
