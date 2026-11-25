"""The exam: run the desk on inputs where the right answer is known.

A research system that finds alpha in pure noise is worse than no system.
Each exam below has a known answer, and the desk has to get it right.
"""

from __future__ import annotations

import math
import shutil
import tempfile
import time

import numpy as np

from . import dsl, ops, pit, stats, synth
from .evaluator import Desk
from .thresholdout import new_salt
from .triallog import TrialLog

REASON = {"counterparty": "investors slow to update", "edge_type": "behavioral", "decay": "months"}


def card(cid, formula, field=None, sign="+", hold=5, family=None, reason=REASON, ref=None):
    c = {"id": cid, "formula": formula, "hold": hold, "reason": reason, "family": family or cid}
    if field:
        c["expect"] = [{"field": field, "sign": sign}]
    if ref:
        c["expect"] = [{"formula": ref, "sign": sign}]
    return c


class Exam:
    def __init__(self, fast: bool = True, seed: int = 7):
        self.fast = fast
        self.seed = seed
        self.tmp = tempfile.mkdtemp(prefix="alphadesk_selftest_")
        n, T = (300, 1500) if fast else (500, 2520)
        self.panel = synth.make_market(n_assets=n, n_days=T, n_noise=40, seed=seed)
        synth.add_leaky_field(self.panel)
        self.results = []

    def desk(self, name, panel=None):
        return Desk(panel or self.panel, TrialLog(f"{self.tmp}/{name}"), library=synth.KNOWN_FACTORS, seed=self.seed)

    def record(self, name, ok, what, detail):
        self.results.append({"exam": name, "ok": bool(ok), "what": what, "detail": detail})

    # ------------------------------------------------------------------
    def noise_1000(self):
        rng = np.random.default_rng(self.seed)
        R = rng.normal(0, 0.01, (252, 1000))      # 1,000 strategies, 1 year, no edge
        srs = np.array([stats.sharpe(R[:, j]) for j in range(R.shape[1])]) * math.sqrt(252)
        best = int(np.argmax(srs))
        survivors = sum(stats.deflated_sr(R[:, j], 1000)["dsr"] >= 0.95 for j in range(R.shape[1]))
        self.record("noise_1000", survivors == 0,
                    "1,000 random strategies, 1 year",
                    f"best naive Sharpe {srs[best]:.2f}, survivors after deflation {survivors}")

    def noise_campaign(self):
        d = self.desk("noise")
        names = [f for f in self.panel.fields if f.startswith("noise_")]
        surv = []
        for f in names:
            v = d.submit(card(f, f"cs_rank({f})", f), campaign="noise")
            if v["survived"]:
                surv.append(f)
        self.record("noise_campaign", not surv, f"{len(names)} noise characteristics through all checks",
                    f"survivors {len(surv)}")

    def planted_found(self):
        d = self.desk("planted")
        got = {}
        for f in ("quality", "flow"):
            v = d.submit(card(f, f"cs_rank({f})", f), campaign="planted")
            got[f] = v["survived"]
        self.record("planted_found", all(got.values()), "planted signals, daily IC 0.03 and 0.02",
                    ", ".join(f"{k} {'found' if v else 'missed'}" for k, v in got.items()))

    def planted_short(self):
        short = synth.make_market(n_assets=60, n_days=520, n_noise=2, seed=self.seed + 1)
        d = self.desk("short", panel=short)
        v = d.submit(card("flow", "cs_rank(flow)", "flow"), campaign="short")
        self.record("planted_short", not v["survived"], "the same IC 0.02 signal on 60 stocks, 2 years",
                    v["summary"])

    def leaking_oracle(self):
        d = self.desk("leak")
        v = d.submit(card("insight", "cs_rank(insight)", "insight"), campaign="leak")
        raw_ic = float(np.nanmean(stats.rowwise_rank_ic(self.panel.get("insight"), self.panel.forward_returns(1))))
        self.record("leaking_oracle", v["died_at"] == 2,
                    "a feature that secretly reads tomorrow's return",
                    f"its IC would be {raw_ic:.2f}; {v['summary']}")

    def shift_ops(self):
        bad = []
        for name, (fn, n_arr, n_num) in ops.OPERATORS.items():
            nums = {0: (), 1: (10,), 2: (-1, 1)}[n_num]
            if name == "neutralize":
                fn2 = lambda x, g: ops.neutralize(x, np.abs(np.round(g * 2)))
                ok = pit.shift_test_operator(fn2, 2, ())
            elif name == "winsorize":
                ok = pit.shift_test_operator(fn, 1, (0.05,))
            elif name == "ewm":
                ok = pit.shift_test_operator(fn, 1, (5,))
            else:
                ok = pit.shift_test_operator(fn, n_arr, nums)
            if not ok:
                bad.append(name)
        self.record("shift_operators", not bad, f"all {len(ops.OPERATORS)} operators, data cut at t vs full",
                    "bit-identical before t" if not bad else f"leaks: {bad}")

    def broken_operator(self):
        def centered_mean(x, w):  # a classic bug: a centered window peeks ahead
            import pandas as pd
            return pd.DataFrame(x).rolling(int(w), center=True, min_periods=1).mean().to_numpy()
        caught = not pit.shift_test_operator(centered_mean, 1, (10,))
        self.record("broken_operator", caught, "a rolling mean with a centered window", "caught" if caught else "missed")

    def mutations(self):
        d = self.desk("mut")
        cases = [("sign flip", "cs_rank(-quality)", 1), ("wrong field", "cs_rank(noise_03)", 1),
                 ("negative lag", "cs_rank(delay(quality, -1))", 2)]
        caught = []
        for name, f, want in cases:
            v = d.submit(card("q_mut", f, "quality", family="q_mut"), campaign="mut")
            caught.append(v["died_at"] == want)
        self.record("code_mutations", all(caught), "quality signal with a flipped sign, a wrong field, a negative lag",
                    f"caught {sum(caught)} of {len(caught)}")

    def rediscover_momentum(self):
        d = self.desk("mom")
        f = "cs_rank(log(delay(close, 21)) - log(delay(close, 252)))"
        v = d.submit(card("trend_follow", f, hold=60, ref="delta(log(close), 231)"), campaign="mom")
        ok = v["died_at"] == 7
        self.record("rediscover_momentum", ok, "12-1 momentum under a new name", v["summary"])

    def bad_ideas(self):
        d = self.desk("bad")
        v1 = d.submit({"id": "no_reason", "formula": "cs_rank(quality)", "hold": 5,
                       "expect": [{"field": "quality", "sign": "+"}]}, campaign="bad")
        v2 = d.submit(card("price_level", "cs_rank(close)", "close"), campaign="bad")
        ok = v1["died_at"] == 0 and not v2["survived"]
        self.record("bad_ideas", ok, "no reason on the card; ranking stocks by price level",
                    f"{v1['summary']} | {v2['summary']}")

    def determinism(self):
        # the validation salt is random per campaign; give both desks the same one
        salt = new_salt()
        da, db = self.desk("det_a"), self.desk("det_b")
        for d in (da, db):
            d.log.write({"kind": "thresholdout", "campaign": "d", "salt": salt})
        a = da.submit(card("quality", "cs_rank(quality)", "quality"), campaign="d")
        b = db.submit(card("quality", "cs_rank(quality)", "quality"), campaign="d")
        same = [c["detail"] for c in a["checks"]] == [c["detail"] for c in b["checks"]]
        self.record("determinism", same, "the same card on two fresh desks", "identical" if same else "differs")

    def _spend_budget(self, d, campaign):
        # dev and validation answers far apart: every query spends budget
        for i in range(10_000):
            if d.validate(campaign, f"probe_{i}", 0.5, -0.5)["budget_left"] <= 0:
                return i + 1
        return None

    def validation_budget(self):
        d = self.desk("budget")
        n = self._spend_budget(d, "vb")
        v = d.submit(card("quality", "cs_rank(quality)", "quality"), campaign="vb")
        c8 = next((c for c in v["checks"] if c["n"] == 8), None)
        ok = n is not None and v["died_at"] == 8 and c8 is not None and c8["detail"] == "validation budget spent"
        self.record("validation_budget", ok, "spend the validation budget, then submit a planted signal",
                    f"budget spent after {n} queries; next card {v['summary']}")

    def validation_restart(self):
        a = self.desk("restart")
        a.validate("vr", "probe_a", 0.5, -0.5)
        b = self.desk("restart")                       # a second desk on the same trial log
        ta, tb = a.thresholdout("vr"), b.thresholdout("vr")
        qb = b.validate("vr", "probe_b", 0.5, -0.5)
        qa = a.validate("vr", "probe_c", 0.5, -0.5)
        salts = {e["salt"] for e in a.log.entries("vr", kind="thresholdout")}
        ok = (ta.salt == tb.salt and len(salts) == 1 and ta.budget == tb.budget
              and qa["budget_left"] == qb["budget_left"] - 1 == ta.total - 3)
        self.record("validation_restart", ok, "two desks on the same trial log",
                    f"one salt ({len(salts)} written), budget left {qa['budget_left']} of {ta.total} after 3 queries")

    def novelty_known(self):
        d = self.desk("lib")
        f = synth.KNOWN_FACTORS["Mom12m"]
        v = d.submit(card("winners_keep_winning", f, hold=60, ref="delta(log(close), 231)"), campaign="lib")
        ok = v["died_at"] == 7
        self.record("known_factor_verbatim", ok, "a factor already in the library, copied as is", v["summary"])

    def pbo_family(self):
        rng = np.random.default_rng(self.seed)
        M = rng.normal(0, 0.01, (1000, 20))
        p = stats.pbo_cscv(M)["pbo"]
        ok = p > 0.3
        self.record("pbo_noise_family", ok, "20 noise variants tuned on the same data", f"PBO {p:.2f}")

    # ------------------------------------------------------------------
    def run(self, on_result=None):
        exams = [self.noise_1000, self.noise_campaign, self.planted_found, self.planted_short,
                 self.leaking_oracle, self.shift_ops, self.broken_operator, self.mutations,
                 self.rediscover_momentum, self.novelty_known, self.bad_ideas, self.pbo_family,
                 self.determinism, self.validation_budget, self.validation_restart]
        for e in exams:
            t0 = time.time()
            e()
            self.results[-1]["seconds"] = round(time.time() - t0, 1)
            if on_result:
                on_result(self.results[-1])
        shutil.rmtree(self.tmp, ignore_errors=True)
        return self.results
