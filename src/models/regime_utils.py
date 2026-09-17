from __future__ import annotations

import logging
import re
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_SIGMA_STATE = re.compile(r"sigma[^\[]*\[(\d+)\]", re.I)


def _param_names_and_values(result):
    params = getattr(result, "params", None)
    values = np.asarray(params, dtype=float).ravel() if params is not None else np.array([])
    names = []
    index = getattr(params, "index", None)
    if index is not None and not callable(index):
        names = [str(x) for x in list(index)]
    if not names or all(n.isdigit() for n in names):
        for obj in (result, getattr(result, "model", None)):
            if obj is None:
                continue
            raw = getattr(obj, "param_names", None)
            if raw:
                names = [str(x) for x in list(raw)]
                break
    return names, values


def _variances_from_param_names(names, values) -> dict:
    variances = {}
    for name, value in zip(names, values):
        if "sigma" not in name.lower():
            continue
        match = _SIGMA_STATE.search(name)
        if match:
            variances[int(match.group(1))] = float(value)
    return variances


def _variances_from_weighted_series(endog: np.ndarray, probs: np.ndarray) -> dict:
    endog = np.asarray(endog, dtype=float).ravel()
    probs = np.asarray(probs, dtype=float)
    if probs.ndim == 1:
        probs = np.column_stack([1.0 - probs, probs])
    n = min(len(endog), len(probs))
    if n == 0 or probs.shape[1] < 2:
        return {}
    endog = endog[:n]
    probs = probs[:n]
    out = {}
    for i in range(probs.shape[1]):
        w = np.clip(probs[:, i], 0.0, None)
        wsum = float(w.sum())
        if wsum <= 1e-8:
            continue
        w = w / wsum
        mu = float(np.dot(w, endog))
        out[i] = float(np.dot(w, (endog - mu) ** 2))
    return out


def identify_high_vol_state(result, endog: Optional[np.ndarray] = None, probs=None) -> int:
    """Return the regime index with the larger variance."""
    names, values = _param_names_and_values(result)
    variances = _variances_from_param_names(names, values)

    if len(variances) < 2 and endog is not None and probs is not None:
        variances = _variances_from_weighted_series(endog, probs)

    if len(variances) < 2 and endog is None:
        model = getattr(result, "model", None)
        if model is not None and hasattr(model, "endog"):
            endog = np.asarray(model.endog, dtype=float).ravel()
            if probs is None:
                probs = getattr(result, "filtered_marginal_probabilities", None)
                if probs is None:
                    probs = getattr(result, "smoothed_marginal_probabilities", None)
            if endog.size and probs is not None:
                variances = _variances_from_weighted_series(endog, probs)

    if variances:
        high = max(variances, key=variances.get)
        logger.info("High-vol regime is state %s (variances=%s)", high, {k: round(v, 4) for k, v in variances.items()})
        return int(high)

    logger.warning("Could not identify regime variances; leaving labels in model order (state 1 = high vol)")
    return 1


def as_prob_array(probs) -> np.ndarray:
    if hasattr(probs, "to_numpy"):
        arr = np.asarray(probs.to_numpy(), dtype=float)
    else:
        arr = np.asarray(probs, dtype=float)
    if arr.ndim == 1:
        arr = np.column_stack([1.0 - arr, arr])
    return arr
