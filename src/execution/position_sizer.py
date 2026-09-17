from __future__ import annotations

from typing import Optional

import numpy as np


class KellyPositionSizer:
    def __init__(
        self,
        max_leverage: float = 1.0,
        half_kelly: bool = True,
        fraction: Optional[float] = None,
    ):
        self.max_leverage = float(max_leverage)
        self.half_kelly = bool(half_kelly)
        if fraction is None:
            fraction = 0.5 if half_kelly else 1.0
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must be in (0, 1]")
        self.fraction = float(fraction)

    def calculate_capital_allocation(self, win_prob: float, win_loss_ratio: float = 1.5) -> float:
        p = float(win_prob)
        if not np.isfinite(p):
            return 0.0
        p = min(max(p, 0.0), 1.0)
        q = 1.0 - p
        b = float(win_loss_ratio)
        if b <= 0:
            return 0.0
        f_star = (p * b - q) / b
        if f_star <= 0:
            return 0.0
        return float(np.clip(f_star * self.fraction, 0.0, self.max_leverage))


if __name__ == "__main__":
    print(KellyPositionSizer().calculate_capital_allocation(0.70))
