"""The evaluator. Plain Python, deterministic, never an agent.

It owns the data windows, runs the checks cheapest-first, writes every
attempt to the trial log and returns a verdict. Agents only ever see the
verdict: passed, or the number of the check that killed the idea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import dsl, pit, portfolio, stats
from .thresholdout import Thresholdout, new_salt
from .triallog import TrialLog

EDGE_TYPES = {"risk_premium", "structural", "behavioral", "informational", "analytical"}

CHECK_NAMES = {
    0: "reason gate", 1: "idea <-> code", 2: "point-in-time", 3: "rank IC",
    4: "decay vs holding", 5: "costs", 6: "microcaps", 7: "novelty",
    8: "subperiods + validation", 9: "deflated Sharpe", 10: "PBO",
    11: "post-cutoff", 12: "holdout",
}


@dataclass
class Config:
    t_min: float = 3.0               # Harvey, Liu, Zhu (2016)
    ic_min: float = 0.02
    ic_leak_audit: float = 0.10      # above this, flag for a leak audit
    decay_keep: float = 0.5          # IC at the holding horizon vs the peak (proposal)
    cost_bps: float = 15.0           # per side
    size_drop: float = 0.3           # drop the smallest 30% for the microcap check
    novelty_max_corr: float = 0.7    # proposal
    novelty_min_alpha_t: float = 2.0
    dsr_min: float = 0.95
    pbo_max: float = 0.5             # proposal
    valid_keep: float = 0.5          # validation IC must keep half of development IC
    horizons: tuple = (1, 5, 10, 20, 60)
    periods_per_year: int = 252


@dataclass
class Windows:
    start: int
    valid_start: int
    cutoff: int
    holdout_start: int
    end: int

    @classmethod
    def from_panel(cls, panel, warmup: int = 260):
        T = panel.T
        m = panel.meta
        return cls(start=min(warmup, T // 5),
                   valid_start=panel.row_of(m["valid_start"]),
                   cutoff=panel.row_of(m["cutoff"]),
                   holdout_start=panel.row_of(m["holdout_start"]),
                   end=T)


@dataclass
class Desk:
    panel: object
    log: TrialLog
    config: Config = field(default_factory=Config)
    library: dict = field(default_factory=dict)  # name -> formula
    external: object = None   # DataFrame of published factor returns (daily or monthly)
    seed: int = 0

    def __post_init__(self):
        self.w = Windows.from_panel(self.panel)
        # one validation budget per campaign, rebuilt from the trial log
        self.tho = {}
        for c in sorted({e.get("campaign") for e in self.log.entries()
                         if e.get("kind") in ("thresholdout", "validation")}):
            self.thresholdout(c, create=False)
        self._lib_cache = None
        self._ret = self.panel.returns()

    # ------------------------------------------------------------ library
    def library_returns(self) -> dict:
        if self._lib_cache is None:
            out = {}
            for name, f in self.library.items():
                sig = dsl.compile_signal(f, self.panel).evaluate(self.panel)
                ls = portfolio.long_short(sig, self._ret, hold=1, start=self.w.start)
                out[name] = {"returns": ls["gross"], "signal": dsl.compile_signal(f, self.panel), "values": sig}
            self._lib_cache = out
        return self._lib_cache

    # ------------------------------------------------------------ validation
    def thresholdout(self, campaign: str, create: bool = True) -> Thresholdout | None:
        """The campaign's Thresholdout, rebuilt from the trial log so every Desk
        on the same log shares one salt and one budget. On first use the salt
        is generated and written to the log."""
        salts = self.log.entries(campaign, kind="thresholdout")
        if salts:
            salt = salts[0]["salt"]
        elif create:
            salt = new_salt()
            self.log.write({"kind": "thresholdout", "campaign": campaign, "salt": salt})
        else:
            return None
        vals = self.log.entries(campaign, kind="validation")
        tho = Thresholdout(salt, used=sum(bool(e.get("overfit_flag")) for e in vals), queries=len(vals))
        self.tho[campaign] = tho
        return tho

    def validate(self, campaign: str, card_id, dev_value: float, valid_value: float) -> dict:
        """Ask the campaign's validation window once and log the query."""
        q = self.thresholdout(campaign).query(dev_value, valid_value)
        self.log.write({"kind": "validation", "campaign": campaign, "card": card_id,
                        "answer": q["value"], "overfit_flag": q["overfit_flag"],
                        **({"exhausted": True} if q.get("exhausted") else {})})
        return q

    # ------------------------------------------------------------ helpers
    def _ic(self, sig, h, a, b):
        fwd = self.panel.forward_returns(h)
        s = sig.copy()
        s[:a] = np.nan
        s[b:] = np.nan
        # a forward return that reaches past b would peek into the next window
        if b - h > a:
            s[b - h : b] = np.nan
        return stats.rowwise_rank_ic(s, fwd)

    def _slice(self, x, a, b):
        y = np.full_like(x, np.nan)
        y[a:b] = x[a:b]
        return y

    # ------------------------------------------------------------ evaluate
    def submit(self, card: dict, campaign: str = "c001") -> dict:
        cfg, w, p = self.config, self.w, self.panel
        checks = []
        verdict = {"card": card.get("id"), "formula": card.get("formula"), "campaign": campaign,
                   "family": card.get("family", card.get("id")), "checks": checks,
                   "survived": False, "died_at": None, "data": p.data_hash()}

        def done(n, status, detail, value=None):
            checks.append({"n": n, "name": CHECK_NAMES[n], "status": status, "detail": detail, "value": value})
            if status == "fail":
                verdict["died_at"] = n
                return True
            return False

        # 0 reason gate --------------------------------------------------
        r = card.get("reason") or {}
        missing = [k for k in ("counterparty", "edge_type", "decay") if not r.get(k)]
        if missing or r.get("edge_type") not in EDGE_TYPES:
            why = f"missing {', '.join(missing)}" if missing else f"edge_type must be one of {sorted(EDGE_TYPES)}"
            done(0, "fail", why)
            return self._finish(verdict, card, campaign, None)
        done(0, "pass", f"{r['edge_type']}: {r['counterparty']}")

        # 2 compile (type check runs before anything touches data) -------
        try:
            sig = dsl.compile_signal(card["formula"], p)
        except dsl.LookAheadError as e:
            done(1, "pass", "skipped, code never compiled")
            done(2, "fail", f"refused to compile: {e}")
            return self._finish(verdict, card, campaign, None)
        except dsl.CompileError as e:
            done(1, "fail", f"does not compile: {e}")
            return self._finish(verdict, card, campaign, None)
        verdict["canon"] = sig.canon
        S = sig.evaluate(p)
        hold = int(card.get("hold", 1))

        # 1 idea <-> code: the behavioral contract written on the card -----
        exp = card.get("expect") or []
        if not exp:
            if done(1, "fail", "the card has no `expect` contract, so the code cannot be checked against the idea"):
                return self._finish(verdict, card, campaign, sig)
        bad = []
        for e in exp:
            ref_name = e.get("field") or e.get("formula")
            try:
                ref = p.get(e["field"]) if e.get("field") else dsl.compile_signal(e["formula"], p).evaluate(p)
            except (KeyError, dsl.CompileError) as err:
                bad.append(f"contract reference `{ref_name}` is invalid: {err}")
                continue
            ic_ref = np.nanmean(stats.rowwise_rank_ic(self._slice(S, w.start, w.valid_start)[::5],
                                                       ref[::5]))
            want = 1 if e.get("sign", "+") == "+" else -1
            if not np.isfinite(ic_ref) or np.sign(ic_ref) != want or abs(ic_ref) < 0.05:
                bad.append(f"expected {e.get('sign', '+')} relation to {ref_name}, code gives {ic_ref:+.2f}")
        if bad:
            if done(1, "fail", "; ".join(bad)):
                return self._finish(verdict, card, campaign, sig)
        else:
            done(1, "pass", "code matches the card's contract")

        # 2 point-in-time -------------------------------------------------
        leaks = [f for f in sorted(sig.fields) if not pit.shift_test_field(p, f)["ok"]]
        if leaks:
            done(2, "fail", f"field {', '.join(leaks)} changes when future rows are removed: its loader reads ahead")
            return self._finish(verdict, card, campaign, sig)
        st_sig = pit.shift_test_signal(sig, p)
        if not st_sig["ok"]:
            done(2, "fail", f"signal changes at rows {st_sig['bad_rows']} when future data is removed")
            return self._finish(verdict, card, campaign, sig)
        done(2, "pass", "type check + shift test on every field")

        # 3 rank IC on the development window -----------------------------
        ic = self._ic(S, hold, w.start, w.valid_start)
        s3 = stats.ic_summary(ic, step=hold)
        verdict["ic_dev"] = s3
        leak_flag = abs(s3["mean"]) > cfg.ic_leak_audit
        if not (s3["t"] >= cfg.t_min and s3["mean"] >= cfg.ic_min):
            done(3, "fail", f"IC {s3['mean']:.3f}, t {s3['t']:.1f} (need t >= {cfg.t_min}, IC >= {cfg.ic_min})", s3)
            return self._finish(verdict, card, campaign, sig, S, hold)
        done(3, "warn" if leak_flag else "pass",
             f"IC {s3['mean']:.3f}, t {s3['t']:.1f}" + ("  · IC above 0.10, audit for a leak" if leak_flag else ""), s3)

        # 4 decay ---------------------------------------------------------
        hz = [h for h in sorted(set(cfg.horizons) | {hold}) if h < (w.valid_start - w.start) // 4]
        by_h = {h: float(np.nanmean(self._ic(S, h, w.start, w.valid_start))) for h in hz}
        verdict["ic_by_horizon"] = by_h
        peak = max(by_h.values())
        if by_h[hold] < cfg.decay_keep * peak:
            best = max(by_h, key=by_h.get)
            done(4, "fail", f"IC at hold {hold} is {by_h[hold]:.3f}, peak {peak:.3f} at {best}: holding period doesn't match the signal", by_h)
            return self._finish(verdict, card, campaign, sig, S, hold)
        done(4, "pass", f"IC at hold {hold}: {by_h[hold]:.3f}, peak {peak:.3f}", by_h)

        # 5 costs ---------------------------------------------------------
        ls = portfolio.long_short(self._slice(S, w.start, w.valid_start), self._ret, hold=hold,
                                  cost_bps=cfg.cost_bps, start=w.start)
        sr_net = stats.sharpe(ls["net"]) * math.sqrt(cfg.periods_per_year)
        sr_gross = stats.sharpe(ls["gross"]) * math.sqrt(cfg.periods_per_year)
        be = portfolio.breakeven_cost_bps(ls["gross"], ls["turnover"])
        verdict["costs"] = {"sharpe_net": sr_net, "sharpe_gross": sr_gross, "breakeven_bps": be,
                            "turnover_per_rebalance": float(np.nanmean(ls["turnover"]))}
        if not sr_net > 0:
            done(5, "fail", f"net Sharpe {sr_net:.2f} at {cfg.cost_bps:g} bps, breakeven {be:.0f} bps")
            return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
        done(5, "pass", f"net Sharpe {sr_net:.2f}, breakeven {be:.0f} bps")

        # 6 microcaps -----------------------------------------------------
        if "size" in p.fields:
            size = p.get("size")
            cut = np.nanquantile(size, cfg.size_drop, axis=1, keepdims=True)
            S_big = np.where(size >= cut, S, np.nan)
            s6 = stats.ic_summary(self._ic(S_big, hold, w.start, w.valid_start), step=hold)
            if not (s6["t"] >= 2.0 and np.sign(s6["mean"]) == np.sign(s3["mean"])):
                done(6, "fail", f"without the smallest {cfg.size_drop:.0%}: IC {s6['mean']:.3f}, t {s6['t']:.1f}", s6)
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
            done(6, "pass", f"without the smallest {cfg.size_drop:.0%}: IC {s6['mean']:.3f}, t {s6['t']:.1f}", s6)
        else:
            done(6, "na", "no size field in this data")

        # 7 novelty -------------------------------------------------------
        lib = self.library_returns()
        if lib:
            names = list(lib)
            y = self._slice(ls["gross"], w.start, w.valid_start)
            X = np.column_stack([lib[n]["returns"] for n in names])
            m = np.isfinite(y) & np.isfinite(X).all(axis=1)
            corr = {n: float(np.corrcoef(y[m], X[m, i])[0, 1]) for i, n in enumerate(names)}
            sims = {n: dsl.ast_similarity(sig, lib[n]["signal"]) for n in names}
            rows = slice(w.start, w.valid_start, 5)
            sig_corr = {n: float(np.nanmean(stats.rowwise_rank_ic(S[rows], lib[n]["values"][rows])))
                        for n in names}
            reg = stats.ols_alpha(y, X)
            verdict["novelty"] = {"corr": corr, "signal_corr": sig_corr, "ast_similarity": sims,
                                  "regression": reg, "factors": names}
            top = max(corr, key=lambda k: abs(corr[k]))
            top_s = max(sig_corr, key=lambda k: abs(sig_corr[k]))
            if abs(sig_corr[top_s]) >= cfg.novelty_max_corr or max(sims.values()) > 0.8:
                done(7, "fail", f"already known: ranks stocks like {top_s} (ρ {sig_corr[top_s]:.2f})", sig_corr)
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
            if abs(corr[top]) >= cfg.novelty_max_corr:
                done(7, "fail", f"already known: returns move with {top} (ρ {corr[top]:.2f})", corr)
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
            if not reg["t_alpha"] >= cfg.novelty_min_alpha_t:
                done(7, "fail", f"explained by known factors: α t {reg['t_alpha']:.1f}", corr)
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
            done(7, "pass", f"max ρ {sig_corr[top_s]:.2f} ({top_s}), α t {reg['t_alpha']:.1f}", sig_corr)
        else:
            done(7, "na", "no factor library loaded")

        # 7b published factors (French, OSAP): returns-based, any frequency ---
        externals = self.external if isinstance(self.external, (list, tuple)) else (
            [self.external] if self.external is not None else [])
        for ext_df in externals:
            ext = self._external_check(ls["gross"], w.start, w.valid_start, ext_df)
            verdict.setdefault("novelty", {}).setdefault("published_all", []).append(ext)
            if ext and "top" in ext:
                verdict["novelty"]["published"] = ext
            if ext and ext.get("max_abs_corr", 0) >= cfg.novelty_max_corr:
                done(7, "fail", f"already published: returns move with {ext['top']} (ρ {ext['top_corr']:.2f})")
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
            if ext and ext.get("t_alpha", 99) < cfg.novelty_min_alpha_t:
                done(7, "fail", f"explained by published factors: α t {ext['t_alpha']:.1f} after {ext['n_factors']} factors")
                return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])

        # 8 subperiods + validation window --------------------------------
        bounds = np.linspace(w.start, w.valid_start, 5).astype(int)
        subs = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            subs.append(stats.ic_summary(self._ic(S, hold, a, b), step=hold))
        signs = [np.sign(x["mean"]) for x in subs]
        best = int(np.argmax([x["mean"] for x in subs]))
        ic_wo_best = np.concatenate([self._ic(S, hold, a, b)[::hold] for i, (a, b) in
                                     enumerate(zip(bounds[:-1], bounds[1:])) if i != best])
        ic_wo_best = ic_wo_best[np.isfinite(ic_wo_best)]
        t_wo = float(ic_wo_best.mean() / (ic_wo_best.std(ddof=1) / math.sqrt(len(ic_wo_best))))
        ic_valid = float(np.nanmean(self._ic(S, hold, w.valid_start, w.cutoff)))
        q = self.validate(campaign, card.get("id"), s3["mean"], ic_valid)
        verdict["subperiods"] = subs
        verdict["validation"] = q
        if q.get("exhausted"):
            done(8, "fail", "validation budget spent")
            return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
        if len(set(signs)) > 1 or t_wo < 2.0:
            done(8, "fail", f"subperiod signs {['+' if s > 0 else '-' for s in signs]}, t without best {t_wo:.1f}")
            return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
        if not q["value"] >= cfg.valid_keep * s3["mean"]:
            done(8, "fail", f"validation window answers IC {q['value']:.3f} vs {s3['mean']:.3f} in development")
            return self._finish(verdict, card, campaign, sig, S, hold, ls["net"])
        done(8, "pass", f"4/4 same sign, t without best {t_wo:.1f}, validation IC {q['value']:.3f}")

        # write the trial before the multiple-testing checks so it counts itself
        self._record(verdict, card, campaign, sig, ls["net"])

        # 9 deflated Sharpe ------------------------------------------------
        M = self.log.returns_matrix(campaign)
        n_eff = stats.effective_n(M) if M is not None else 1.0
        n_raw = M.shape[1] if M is not None else 1
        # benchmark: the expected best Sharpe of N_eff strategies with no edge.
        # Using the spread of the trials' own Sharpes would punish a real
        # signal for sharing a campaign with another real signal.
        d = stats.deflated_sr(ls["net"], max(n_eff, 1.0))
        sr_ann = d["sr"] * math.sqrt(cfg.periods_per_year)
        minbtl = stats.min_backtest_length_years(max(n_eff, 1.0), sr_ann) if sr_ann > 0 else float("inf")
        verdict["dsr"] = {**d, "n_raw": n_raw, "n_eff": n_eff, "sr_annual": sr_ann, "minbtl_years": minbtl}
        if not d["dsr"] >= cfg.dsr_min:
            done(9, "fail", f"DSR {d['dsr']:.2f} at N_eff {n_eff:.0f} (raw {n_raw})", d["dsr"])
            return self._finish(verdict, card, campaign, sig, S, hold, recorded=True)
        done(9, "pass", f"DSR {d['dsr']:.2f} at N_eff {n_eff:.0f} (raw {n_raw}), MinBTL {minbtl:.1f}y", d["dsr"])

        # 10 PBO over the family ------------------------------------------
        fam = self.log.returns_matrix(campaign, family=verdict["family"])
        if fam is None or fam.shape[1] < 4:
            done(10, "na", f"family has {0 if fam is None else fam.shape[1]} variants, PBO needs 4+")
        else:
            pb = stats.pbo_cscv(fam[w.start:])
            verdict["pbo"] = pb
            if not pb["pbo"] < cfg.pbo_max:
                done(10, "fail", f"PBO {pb['pbo']:.2f} over {fam.shape[1]} variants", pb["pbo"])
                return self._finish(verdict, card, campaign, sig, S, hold, recorded=True)
            done(10, "pass", f"PBO {pb['pbo']:.2f} over {fam.shape[1]} variants", pb["pbo"])

        # 11 post-cutoff ---------------------------------------------------
        ic_post = self._ic(S, hold, w.cutoff, w.holdout_start)
        s11 = stats.ic_summary(ic_post, step=hold)
        ls_post = portfolio.long_short(self._slice(S, w.cutoff, w.holdout_start), self._ret, hold=hold,
                                       cost_bps=cfg.cost_bps, start=w.cutoff)
        d_post = stats.deflated_sr(ls_post["net"], 1)
        need = stats.min_track_record(d["sr"], 0.0, d["skew"], d["kurt"])
        have = int(np.isfinite(ls_post["net"]).sum())
        verdict["post_cutoff"] = {"ic": s11, "sharpe_annual": d_post["sr"] * math.sqrt(cfg.periods_per_year),
                                  "rows": have, "rows_needed": need}
        if not np.sign(s11["mean"]) == np.sign(s3["mean"]):
            done(11, "fail", f"after the model cutoff the sign flips: IC {s11['mean']:.3f}")
            return self._finish(verdict, card, campaign, sig, S, hold, recorded=True)
        if have < need:
            done(11, "weak", f"same sign (IC {s11['mean']:.3f}), but {have} rows prove the sign, not the size; "
                             f"{need:.0f} needed")
        else:
            done(11, "pass", f"IC {s11['mean']:.3f} on {have} rows after the cutoff")

        done(12, "sealed", "run `alphadesk holdout` yourself, once")
        verdict["survived"] = True
        return self._finish(verdict, card, campaign, sig, recorded=True)

    def _external_check(self, gross, a, b, ext, max_factors: int = 12):
        import pandas as pd
        r = pd.Series(gross[a:b], index=self.panel.dates[a:b]).dropna()
        if r.empty:
            return None
        monthly = (ext.index.to_series().diff().dt.days.median() or 1) > 20
        if monthly:
            r = (1 + r).groupby(r.index.to_period("M")).prod() - 1
            r.index = r.index.to_timestamp("M")
            e = ext.copy()
            e.index = e.index.to_period("M").to_timestamp("M")
        else:
            e = ext
        j = pd.concat([r.rename("_c"), e], axis=1, join="inner").dropna(axis=1, thresh=int(0.8 * len(r)))
        j = j.dropna()
        if len(j) < 24:
            return {"skipped": f"only {len(j)} overlapping periods with the published factors"}
        corr = j.drop(columns="_c").corrwith(j["_c"]).dropna()
        top = corr.abs().idxmax()
        picks = list(corr.abs().sort_values(ascending=False).index[:max_factors])
        reg = stats.ols_alpha(j["_c"].to_numpy(), j[picks].to_numpy())
        return {"top": top, "top_corr": float(corr[top]), "max_abs_corr": float(abs(corr[top])),
                "t_alpha": reg["t_alpha"], "n_factors": len(picks), "periods": int(len(j)),
                "frequency": "monthly" if monthly else "daily"}

    # ------------------------------------------------------------ logging
    def _record(self, verdict, card, campaign, sig, net):
        if verdict.get("_recorded"):
            return
        self.log.write({"kind": "trial", "campaign": campaign, "card": card.get("id"),
                        "family": verdict["family"], "formula": card.get("formula"),
                        "canon": sig.canon if sig else None, "data": verdict["data"]},
                       returns=net)
        verdict["_recorded"] = True

    def _finish(self, verdict, card, campaign, sig, S=None, hold=1, net=None, recorded=False):
        if not recorded and not verdict.get("_recorded"):
            if net is None and S is not None:
                net = portfolio.long_short(self._slice(S, self.w.start, self.w.valid_start), self._ret,
                                           hold=hold, cost_bps=self.config.cost_bps, start=self.w.start)["net"]
            self._record(verdict, card, campaign, sig, net)
        verdict.pop("_recorded", None)
        last = verdict["checks"][-1] if verdict["checks"] else None
        verdict["summary"] = ("survived" if verdict["survived"]
                              else f"died at {last['n']} {last['name']}: {last['detail']}" if last else "no checks")
        self.log.write({"kind": "verdict", "campaign": campaign, "card": card.get("id"),
                        "family": verdict["family"], "formula": card.get("formula"),
                        "survived": verdict["survived"], "died_at": verdict["died_at"],
                        "summary": verdict["summary"] if not verdict["survived"] else "survived"})
        # keep the full verdict and the card for reports (the evaluator's side of the wall)
        import json as _json
        vdir = self.log.root / "verdicts" / campaign
        vdir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(card.get("id", "card")))
        (vdir / f"{safe}.json").write_text(_json.dumps({"card": card, "verdict": verdict}, default=float),
                                           encoding="utf-8")
        return verdict
