"""Reject accidentally importing an editable installation outside this release."""

import importlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pytest_sessionstart(session):
    for name in ("contracts", "value_model", "evaluate", "policy", "pipeline", "synthetic"):
        module = importlib.import_module(name)
        if Path(module.__file__).resolve().parent != ROOT:
            raise RuntimeError(f"Foreign import shadow for {name}")
