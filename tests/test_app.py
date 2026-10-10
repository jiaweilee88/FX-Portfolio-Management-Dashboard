from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.runtime.caching import cache_utils
from streamlit.testing.v1 import AppTest

from src.data import market_data as md

APP = str(Path(__file__).resolve().parent.parent / "app.py")


class FakeClock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _fake_yf_history(symbol, **kwargs):
    today = date.today()
    if "period" in kwargs:
        # CNH=X returns a single row live, so USD/CNH has no previous close
        index = pd.bdate_range(end=today, periods=1 if symbol == "CNH=X" else 2)
    else:
        index = pd.bdate_range(end=today - timedelta(days=1), periods=300)
    seed = sum(map(ord, symbol))
    values = np.exp(np.cumsum(np.random.default_rng(seed).normal(0, 0.005, len(index))))
    return pd.Series(values, index=pd.DatetimeIndex(index, name="date"), name=symbol)


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    clock = FakeClock()
    calls = {"history": 0, "current": 0}
    real_history, real_current = md.load_history, md.load_current_quotes

    def counted_history(*args, **kwargs):
        calls["history"] += 1
        return real_history(*args, **kwargs)

    def counted_current(*args, **kwargs):
        calls["current"] += 1
        return real_current(*args, **kwargs)

    monkeypatch.setattr(md, "_yf_history", _fake_yf_history)
    monkeypatch.setattr(md, "DEFAULT_CACHE_DIR", tmp_path)
    monkeypatch.setattr(md, "load_history", counted_history)
    monkeypatch.setattr(md, "load_current_quotes", counted_current)
    monkeypatch.setattr(cache_utils, "TTLCACHE_TIMER", clock)
    st.cache_data.clear()
    yield clock, calls
    st.cache_data.clear()


def _run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def test_app_renders_a_snapshot(app_env):
    at = _run()

    labels = [m.label for m in at.metric]
    assert "Inception P&L (USD)" in labels and "Daily P&L (USD)" in labels
    assert any("T0018" in w.value for w in at.warning)  # deliberately invalid sample row
    assert len(at.dataframe) == 1


def test_daily_total_carries_footnote_for_blank_positions(app_env):
    at = _run()

    daily = next(m for m in at.metric if m.label == "Daily P&L (USD)")
    assert daily.value != "n/a"  # the blank position didn't blank the total
    captions = [c.value for c in at.caption]
    assert any(
        "Excludes 1 of 17 positions with blank daily P&L" in c
        and "missing its previous close (data gap): T0009" in c
        for c in captions
    ), captions


def test_current_quotes_cached_60s_history_cached_24h(app_env):
    clock, calls = app_env

    _run()
    assert calls == {"history": 1, "current": 1}

    clock.advance(30)  # rerun within 60s: both served from cache
    _run()
    assert calls == {"history": 1, "current": 1}

    clock.advance(31)  # 61s: quotes refetched, history still cached
    _run()
    assert calls == {"history": 1, "current": 2}

    clock.advance(24 * 60 * 60)  # past 24h: history refetched too
    _run()
    assert calls == {"history": 2, "current": 3}
