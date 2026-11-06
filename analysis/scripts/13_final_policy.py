"""Final heterogeneity, recast scenario and honest targeting evaluation.

Inputs: results/11_units_lgbm_{P}_{log,abs}.parquet (10x cross-fitted MOSAIC-DR nuisances).
- CATE (log scale) is learned only inside the joint support and transported elsewhere (flagged).
- Recast totals are split into the in-support part and the transported part.
- In-support totals are cross-checked with the absolute-scale DR policy mean (no retransformation).
- Targeting: cross-fitted CATE ranking vs size ranking vs random, evaluated with held-out
  absolute-scale DR scores (unbiased for any fixed targeting rule) and log-scale TOC/AUTOC.
"""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold
from common import load, RES
from causal_core import add_context, PLANT_X, weighted_se, cluster_se, joint_support, labelled_country

EF = {"N": 1 - (1 - 0.80) / (1 - 0.55), "P": 1 - (1 - 0.90) / (1 - 0.60)}
B = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def cate_model(seed=0):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=7, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                             verbose=-1, random_state=seed, n_jobs=8)


def fit_cate(X, phi, h, sup, seed=0):
    """Out-of-fold CATE for support units; fold-averaged transport for all others."""
    phi = np.clip(phi, *np.quantile(phi[sup], [0.01, 0.99]))
    tau = np.zeros(len(phi)); tau_out = np.zeros(len(phi)); si = np.where(sup)[0]
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(si):
        m = cate_model(seed).fit(X[si[tr]], phi[si[tr]], sample_weight=h[si[tr]])
        tau[si[te]] = m.predict(X[si[te]]); tau_out += m.predict(X) / 5
    tau[~sup] = tau_out[~sup]
    return tau


def wls_cluster(y, Z, w, c):
    Zw = Z * np.sqrt(w)[:, None]
    beta = np.linalg.lstsq(Zw, y * np.sqrt(w), rcond=None)[0]
    u = (y - Z @ beta) * w
    bread = np.linalg.inv(Z.T @ (Z * w[:, None]))
    meat = sum(np.outer(Z[c == g].T @ u[c == g], Z[c == g].T @ u[c == g]) for g in np.unique(c))
    return beta, np.sqrt(np.diag(bread @ meat @ bread))


summary, blp_rows, gates_rows, target_rows, country_rows, toc_rows = [], [], [], [], [], []
for P in ["N", "P"]:
    other = "f_PRemoval" if P == "N" else "f_NRemoval"
    u = pd.read_parquet(RES / f"11_units_lgbm_{P}_log.parquet")
    ua = pd.read_parquet(RES / f"11_units_lgbm_{P}_abs.parquet")
    assert (u.uwwCode.values == ua.uwwCode.values).all()
    _, ctx = add_context(load(P))
    X_cols = PLANT_X + [other] + ctx
    X = u[X_cols].astype(float).to_numpy()
    e = u.e.to_numpy(); h = e * (1 - e); c = u.country.values
    sup = joint_support(e, u.pi1, u.pi0, labelled_country(u))
    phi = u.phi.to_numpy(); phi_abs = ua.phi.to_numpy()
    T0 = (u["T"] == 0).to_numpy(); load_pe = u.load_pe.to_numpy()

    tau = fit_cate(X, phi, h, sup)
    # BLP and GATES (support only)
    mods = {"size (log10 p.e.)": u.log_load, "utilisation": u.util, "latitude (10 deg)": u.lat / 10,
            "requirement in force": u[f"req_{'NTot' if P == 'N' else 'PTot'}"].fillna(0),
            f"co-installed {'P' if P == 'N' else 'N'} removal": u[other],
            "hydraulic load (log10 m3/p.e.)": u.ww_per_pe.fillna(u.ww_per_pe.median())}
    Z = np.column_stack([np.ones(len(u))] + [(m - np.average(m[sup], weights=h[sup])).to_numpy() for m in mods.values()])
    beta, se = wls_cluster(phi[sup], Z[sup], h[sup], c[sup])
    for nm, b_, s_ in zip(["intercept (ATO)"] + list(mods), beta, se):
        blp_rows.append(dict(P=P, term=nm, coef=b_, se=s_, z=b_ / s_))
    q = np.full(len(u), -1); q[sup] = pd.qcut(tau[sup], 5, labels=False)
    for g in range(5):
        th, s_ = weighted_se(phi, h, c, (q == g) & sup)
        gates_rows.append(dict(P=P, group=g + 1, pred_mean=tau[(q == g) & sup].mean(), dr=th, se=s_))
    # heterogeneity test: top vs bottom quintile difference
    g1 = gates_rows[-5]; g5 = gates_rows[-1]
    gates_rows.append(dict(P=P, group="Q1-Q5", dr=g1["dr"] - g5["dr"], se=np.hypot(g1["se"], g5["se"])))

    # baseline per-capita discharge (measurement-standardised; smearing for model-based values)
    obs0 = (u.R == 1).to_numpy() & T0
    smear = np.mean(10 ** (u.Y - u.off - u.mu0)[obs0])
    q0 = np.where(obs0, 10 ** (u.Y - u.off), smear * 10 ** u.mu0)
    base_t = load_pe * q0 / 1000
    red_t = base_t * (1 - 10 ** tau)
    for scope, thr in [(">=10k p.e.", 1e4), (">=150k p.e.", 1.5e5)]:
        pol = (u.capacity_pe >= thr).to_numpy() & T0
        rng = np.random.default_rng(0); cs = np.unique(c); boot, boot_in = [], []
        for b in range(B):
            pick = rng.choice(cs, len(cs), replace=True)
            idx = np.concatenate([np.where((c == k) & sup)[0] for k in pick])
            ph = np.clip(phi[idx], *np.quantile(phi[sup], [0.01, 0.99]))
            m = cate_model(b).fit(X[idx], ph, sample_weight=h[idx])
            r_b = base_t[pol] * (1 - 10 ** m.predict(X[pol]))
            boot.append(r_b.sum()); boot_in.append(r_b[sup[pol]].sum())
        # absolute-scale DR cross-check inside support: sum of load x DR score (t/yr)
        ps = pol & sup
        dr_abs_tot = -np.sum(load_pe[ps] * phi_abs[ps]) / 1000
        s_ = pd.Series(-load_pe[ps] * phi_abs[ps] / 1000).groupby(c[ps]).sum()
        dr_abs_se = np.sqrt(((s_ - s_.sum() / len(s_)) ** 2).sum() * len(s_) / max(len(s_) - 1, 1))
        summary.append(dict(P=P, scope=scope, n_plants=int(pol.sum()), n_in_support=int(ps.sum()),
                            load_pe_M=load_pe[pol].sum() / 1e6, load_pe_M_in_support=load_pe[ps].sum() / 1e6,
                            baseline_t=base_t[pol].sum(), baseline_t_in_support=base_t[ps].sum(),
                            red_total_t=red_t[pol].sum(), red_total_lo=np.quantile(boot, 0.025), red_total_hi=np.quantile(boot, 0.975),
                            red_in_support_t=red_t[ps].sum(), red_in_support_lo=np.quantile(boot_in, 0.025),
                            red_in_support_hi=np.quantile(boot_in, 0.975),
                            red_transported_t=red_t[pol & ~sup].sum(),
                            DRabs_in_support_t=dr_abs_tot, DRabs_in_support_se=dr_abs_se,
                            EF_total_t=EF[P] * base_t[pol].sum(), EF_in_support_t=EF[P] * base_t[ps].sum()))
        print(summary[-1], flush=True)
        if scope == ">=10k p.e.":
            country_rows.append(pd.DataFrame({"country": c[pol], "in_support": sup[pol], "base_t": base_t[pol],
                                              "red_t": red_t[pol], "ef_t": EF[P] * base_t[pol]}).assign(P=P))

    # honest targeting evaluation among untreated support plants >= 2k p.e.
    cand = sup & T0 & (u.capacity_pe >= 2e3).to_numpy()
    tau_abs = fit_cate(X, phi_abs, h, sup, seed=1)          # cross-fitted, kg/p.e./yr
    score = -load_pe * phi_abs / 1000                         # DR estimate of plant reduction, t/yr
    rules = {"CATE-informed": -load_pe * tau_abs / 1000, "Size (p.e.)": load_pe,
             "Random": np.random.default_rng(7).random(len(u))}
    ci = np.where(cand)[0]
    for budget in [0.05, 0.10, 0.25, 0.50, 1.0]:
        k = max(1, int(round(budget * len(ci))))
        vals = {}
        for rname, r in rules.items():
            top = ci[np.argsort(-r[ci])[:k]]
            vals[rname] = top
        for rname, top in vals.items():
            # country-cluster SE of the DR total
            s_ = pd.Series(score[top]).groupby(c[top]).sum()
            target_rows.append(dict(P=P, budget=budget, rule=rname, n=k, DR_reduction_t=score[top].sum(),
                                    se=np.sqrt((s_ ** 2).sum() - s_.sum() ** 2 / max(len(s_), 1)) if len(s_) > 1 else np.nan,
                                    predicted_t=(-load_pe[top] * tau_abs[top] / 1000).sum()))
        diff = score[vals["CATE-informed"]].sum() - score[vals["Size (p.e.)"]].sum()
        rng = np.random.default_rng(3); cs = np.unique(c[ci]); bd = []
        for b in range(B):
            pick = rng.choice(cs, len(cs), replace=True)
            idx = np.concatenate([ci[c[ci] == kk] for kk in pick])
            kb = max(1, int(round(budget * len(idx))))
            a1 = idx[np.argsort(-rules["CATE-informed"][idx])[:kb]]
            a2 = idx[np.argsort(-rules["Size (p.e.)"][idx])[:kb]]
            bd.append(score[a1].sum() - score[a2].sum())
        target_rows.append(dict(P=P, budget=budget, rule="CATE minus size", n=k, DR_reduction_t=diff,
                                lo=np.quantile(bd, 0.025), hi=np.quantile(bd, 0.975)))
    # log-scale TOC / AUTOC (relative effect) for CATE vs size ranking on support
    si = np.where(sup)[0]; base_mean = np.average(phi[si], weights=h[si])
    for rname, r in [("CATE-informed", -tau), ("Size (p.e.)", load_pe)]:
        order = si[np.argsort(-r[si])]
        qs = np.linspace(0.05, 1, 20); toc = []
        for qq in qs:
            top = order[:max(1, int(qq * len(order)))]
            toc.append(np.average(phi[top], weights=h[top]) - base_mean)
        for qq, t in zip(qs, toc):
            toc_rows.append(dict(P=P, rule=rname, q=qq, TOC=t))
        toc_rows.append(dict(P=P, rule=rname, q="AUTOC", TOC=np.mean(toc)))
    u.assign(tau=tau, tau_abs=tau_abs, in_support=sup, base_t=base_t, red_t=red_t)[
        ["uwwCode", "country", "lat", "lon", "load_pe", "capacity_pe", "T", "R", "e", "in_support", "tau", "tau_abs",
         "base_t", "red_t"]].to_parquet(RES / f"13_plant_effects_{P}.parquet", index=False)

pd.set_option("display.width", 260)
for nm, rows in [("summary", summary), ("blp", blp_rows), ("gates", gates_rows), ("targeting", target_rows), ("toc", toc_rows)]:
    df = pd.DataFrame(rows); df.to_csv(RES / f"13_{nm}.csv", index=False)
    if nm != "toc":
        print(df.round(3).to_string())
pd.concat(country_rows).to_csv(RES / "13_country_reductions.csv", index=False)

# absolute-scale DR cross-check over supported target plants: load-weighted sum of DR scores, with a
# country-clustered SE of the score residuals around the cross-fitted absolute-scale predictions
xrows = []
for P in ["N", "P"]:
    ua = pd.read_parquet(RES / f"11_units_lgbm_{P}_abs.parquet")
    pe = pd.read_parquet(RES / f"13_plant_effects_{P}.parquet")
    assert (ua.uwwCode.values == pe.uwwCode.values).all()
    for scope, thr in [(">=10k p.e.", 1e4), (">=150k p.e.", 1.5e5)]:
        m = (pe.capacity_pe >= thr).values & (pe["T"] == 0).values & pe.in_support.values
        x = -pe.load_pe.values[m] * ua.phi.values[m] / 1000
        xh = -pe.load_pe.values[m] * pe.tau_abs.values[m] / 1000
        s_ = pd.Series(x - xh).groupby(pe.country.values[m]).sum()
        G = len(s_); se_ = np.sqrt((s_ ** 2).sum() * G / max(G - 1, 1))
        xrows.append(dict(P=P, scope=scope, n=int(m.sum()), DRabs_total_t=x.sum(), se_cluster_resid=se_,
                          lo=x.sum() - 1.96 * se_, hi=x.sum() + 1.96 * se_))
pd.DataFrame(xrows).to_csv(RES / "13_drabs_crosscheck.csv", index=False)
print(pd.DataFrame(xrows).round(1).to_string())
