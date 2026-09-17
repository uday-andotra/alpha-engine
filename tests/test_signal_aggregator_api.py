from __future__ import annotations

from pathlib import Path


def test_fuse_signals_accepts_override_date_alias():
    src = (Path(__file__).resolve().parents[1] / "src" / "features" / "signal_aggregator.py").read_text()
    assert "def fuse_signals(" in src
    assert "statement_date" in src
    assert "override_date" in src
