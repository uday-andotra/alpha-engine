from __future__ import annotations

from datetime import date

import numpy as np

from data_pipeline.dates import parse_date_from_filename, parse_date_from_text
from features.vol_math import calibrate_ou_ols, garman_klass_variance
from models.llm_sentiment import parse_llm_json


def test_garman_klass_math():
    gk = garman_klass_variance(
        np.array([100.0]),
        np.array([105.0]),
        np.array([98.0]),
        np.array([102.0]),
    )
    assert float(gk[0]) > 0.0


def test_garman_klass_clips_negatives():
    # Pathological OHLC can produce a tiny negative raw GK term
    gk = garman_klass_variance(
        np.array([100.0]),
        np.array([100.01]),
        np.array([99.99]),
        np.array([90.0]),
    )
    assert gk[0] >= 0.0


def test_ou_mean_reversion_recovers_mu():
    rng = np.random.default_rng(0)
    theta, mu, sigma, dt = 5.0, 15.0, 2.0, 1.0 / 252.0
    n = 3000
    x = np.empty(n)
    x[0] = mu
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * rng.normal()
    params = calibrate_ou_ols(x, dt=dt)
    assert abs(params["mu"] - mu) < 1.5
    assert params["theta"] > 0
    assert params["stat_std"] > 0
    assert abs(params["stat_std"] - sigma / np.sqrt(2 * theta)) / (sigma / np.sqrt(2 * theta)) < 0.35


def test_parse_mpc_dates():
    assert parse_date_from_text("Reserve Bank of India February 8, 2024 Monetary Policy") == date(2024, 2, 8)
    assert parse_date_from_filename("MPC_2024-06-07.pdf") == date(2024, 6, 7)
    assert parse_date_from_filename("MPC-2026-05-08.pdf") == date(2026, 5, 8)


def test_parse_llm_json_fences():
    raw = '```json\n{"sentiment_score": 0.25, "reasoning": "hawkish"}\n```'
    parsed = parse_llm_json(raw)
    assert parsed["sentiment_score"] == 0.25
