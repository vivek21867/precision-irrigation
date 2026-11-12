"""Headless smoke tests: every Streamlit page renders against real pipeline outputs."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app/main.py", "app/dashboard.py", *sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "app/pages").glob("*.py"))]
READY = (ROOT / "models/transformer.pt").exists() and (ROOT / "results/metrics/test_metrics_table.csv").exists()


@pytest.mark.skipif(not READY, reason="models/results not generated yet")
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.chdir(ROOT)
    monkeypatch.syspath_prepend(str(ROOT / "app"))
    at = AppTest.from_file(str(ROOT / page), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
