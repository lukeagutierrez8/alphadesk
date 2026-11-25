"""Reusable validation window.

Thresholdout, from Dwork, Feldman, Hardt, Pitassi, Reingold, Roth,
"The reusable holdout: Preserving validity in adaptive data analysis",
Science 349(6248):636-638, 2015.

If the validation value is close to the development value, the agent gets
the development value back and learns nothing new about the validation
window. Only when they disagree by more than a noisy threshold does it get a
noisy validation value. That is what lets agents ask the validation window
many times without turning it into training data.
"""

from __future__ import annotations

import secrets

import numpy as np

# tags that keep the two noise streams apart
_T_HAT, _QUERY = 0, 1


def new_salt() -> str:
    """A fresh campaign salt. Never derived from a fixed seed."""
    return secrets.token_hex(16)


class Thresholdout:
    """One campaign's validation window.

    All noise is drawn from (salt, counter): the noisy threshold from
    (salt, number of budget-spending answers so far), each query's noise from
    (salt, query number). So the state can be rebuilt from the trial log alone:
    ``queries`` validation entries were written, ``used`` of them spent budget.
    """

    def __init__(self, salt: str, threshold: float = 0.01, sigma: float = 0.004, budget: int = 200,
                 used: int = 0, queries: int = 0):
        self.salt = salt
        self.threshold = threshold
        self.sigma = sigma
        self.total = budget
        self.used = used
        self.budget = budget - used
        self.queries = queries
        self._t_hat = self.threshold + self._rng(_T_HAT, self.used).laplace(scale=2 * sigma)

    def _rng(self, tag: int, n: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([int(self.salt, 16), tag, n]))

    def query(self, dev_value: float, valid_value: float) -> dict:
        self.queries += 1
        if self.budget <= 0:
            return {"value": float("nan"), "overfit_flag": False, "budget_left": 0, "exhausted": True}
        if not (np.isfinite(dev_value) and np.isfinite(valid_value)):
            return {"value": float("nan"), "overfit_flag": False, "budget_left": self.budget}
        rng = self._rng(_QUERY, self.queries)
        gap = abs(dev_value - valid_value)
        if gap > self._t_hat + rng.laplace(scale=4 * self.sigma):
            self.budget -= 1
            self.used += 1
            self._t_hat = self.threshold + self._rng(_T_HAT, self.used).laplace(scale=2 * self.sigma)
            return {"value": float(valid_value + rng.laplace(scale=self.sigma)),
                    "overfit_flag": True, "budget_left": self.budget}
        return {"value": float(dev_value), "overfit_flag": False, "budget_left": self.budget}
