"""Rung 3: self-checks, nominal run, Monte Carlo and the frozen prediction file."""
import os, sys, json, time
import numpy as np
import smc_rom as S

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
FT2M = 0.3048


def load_hinge():
    p = os.path.join(RES, "r3_flange_hinge.json")
    if os.path.exists(p):
        d = json.load(open(p)); S.HINGE_TH = np.array(d["theta_p"]); S.HINGE_M = np.array(d["M"])
        return d
    return None


def time_points(A, W, n):
    """T1..T10 of the packet (s from pulse start) for the pulse a = A hav(t/W)^n."""
    x = np.linspace(0, 1, 200001); s = (0.5 * (1 - np.cos(2 * np.pi * x))) ** n
    ip = np.argmax(s)
    up = lambda f: x[np.argmax(s[:ip] >= f)] * W
    dn = lambda f: x[ip + np.argmax(s[ip:] <= f)] * W
    T6 = dn(0.25)
    return [up(0.5), up(0.75), x[ip] * W, dn(0.75), dn(0.5), T6, T6 + 0.25e-3, T6 + 0.5e-3, T6 + 1.0e-3, T6 + 1.5e-3]


def point_at(h, k, i):
    """relative displacement (mm) of points B and C with respect to A at history index k for
    simulation i; packet axes (X right, Y down, Z into the block)."""
    yB, yC = -1.26 * np.cos(np.deg2rad(30)), 0.35
    VB = h["v0"][k, i] + S.T_FL * h["phif"][k, i]; WB = -(h["w0"][k, i] - yB * h["phif"][k, i])
    VC = h["vt"][k, i]; WC = -(h["wt"][k, i] - yC * h["phit"][k, i])
    return float(25.4 * VB), float(25.4 * WB), float(25.4 * VC), float(25.4 * WC)


def selfchecks():
    out = {}
    c = json.load(open(os.path.join(RES, "r3_selfcheck_ccx.json")))
    f_rom = S.clamped_frequency()[0]; f_ccx = c["ccx_freq_hz"][0]
    out["3a"] = dict(rom_hz=float(f_rom), ccx_hz=f_ccx, err=float(f_rom / f_ccx - 1), tol=0.05, passed=bool(abs(f_rom / f_ccx - 1) <= 0.05),
                     rom_hz_26el=float(S.clamped_frequency(26)[0]))
    # 3b: static, elastic, no preload/clearance/friction, 1000 g -- as registered (rigid flange) and after recalibration
    tgt_tip, tgt_T1 = abs(c["static_ccx"]["tip_uy"]), c["static_ccx"]["bolts"][0]["T_full_bolt"]
    def stat(**kw):
        P = S.params(1, F_pre=0.0, mu=0.0, **kw); P["clear"][:] = 0; P["clear_nom"][:] = 0
        o = S.simulate(S.TESTS[:1, 1:], P, linear=True, static_acc=np.array([1000.0]), t_end=0.03)
        return float(o["hist"]["vt"][-1][0]), float(o["hist"]["T"][-1][0, 0])
    tip0, T0 = stat(k_f=1e30, r_c=S.D_FL / 2); tip1, T1 = stat()
    out["3b"] = dict(ccx_tip=tgt_tip, ccx_T1=tgt_T1, registered_model=dict(tip=tip0, T1=T0, err_tip=tip0 / tgt_tip - 1, err_T1=T0 / tgt_T1 - 1),
                     recalibrated=dict(tip=tip1, T1=T1, err_tip=tip1 / tgt_tip - 1, err_T1=T1 / tgt_T1 - 1, k_f=S.NOMINAL["k_f"], r_c=S.NOMINAL["r_c"]),
                     tol=0.15, passed_as_registered=bool(abs(tip0 / tgt_tip - 1) <= 0.15 and abs(T0 / tgt_T1 - 1) <= 0.15))
    # 3c energy balance at the most severe test (by peak g x dV: test 16) and at test 19
    o = S.simulate(S.TESTS[:, 1:], S.params(19)); e = o["energy"]
    bal = (e["Wext"] - e["KE"] - e["dE"] - e["Wpl"] - e["Wfr"] - e["Wdamp"] - e["Wfrac"] - e["Wh"]) / e["Wext"]
    out["3c"] = dict(balance_all_tests=[float(b) for b in bal], worst=float(np.abs(bal).max()), tol=0.02, passed=bool(np.abs(bal).max() <= 0.02))
    # 3d time-step halving, test nearest to first fracture without fracture in the nominal run
    nb = o["broken"].sum(1); k1 = o["kappa"][:, 0]
    cand = np.where(nb == 0)[0]; i = cand[np.argmax(k1[cand])] if len(cand) else 0
    a = S.simulate(S.TESTS[i:i + 1, 1:], S.params(1), dt=2.5e-7); b = S.simulate(S.TESTS[i:i + 1, 1:], S.params(1), dt=1.25e-7)
    e1, e2 = float(a["peak"]["ep"][0, 0]), float(b["peak"]["ep"][0, 0])
    out["3d"] = dict(test=int(S.TESTS[i, 0]), ep1_dt=e1, ep1_half_dt=e2, change=float(abs(e2 / e1 - 1)) if e1 > 0 else 0.0, tol=0.02,
                     passed=bool(e1 > 0 and abs(e2 / e1 - 1) <= 0.02))
    return out, o


def monte_carlo(N=400, seed=20261115):
    rng = np.random.default_rng(seed)
    nt = len(S.TESTS)
    P1 = S.params(N, rng)
    P1["k_f"] = S.NOMINAL["k_f"] * rng.uniform(0.8, 1.25, N); P1["r_c"] = rng.uniform(1.0, 1.25, N); P1["fM"] = rng.uniform(0.85, 1.15, N)
    P = {k: np.repeat(v, nt, axis=0) for k, v in P1.items()}        # same parameter draw for all 19 tests
    tests = np.tile(S.TESTS[:, 1:], (N, 1))
    o = S.simulate(tests, P, sample_dt=5e-5)
    nb = o["broken"].sum(1).reshape(N, nt)
    return P1, nb, o["broken"].reshape(N, nt, 6), o["t_break"].reshape(N, nt, 6)


if __name__ == "__main__":
    hinge = load_hinge()
    t0 = time.time()
    checks, nom = selfchecks()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "balance_all_tests"} for k, v in checks.items()}, indent=1))
    dv = S.TESTS[:, 1] * FT2M
    P1, nb, brk, tbr = monte_carlo(int(sys.argv[1]) if len(sys.argv) > 1 else 400)
    N = len(nb)
    def first(mask):       # min dV over tests where mask true, per sample (inf if none)
        return np.array([dv[m].min() if m.any() else np.inf for m in mask])
    q1 = first(nb >= 1); q2 = first(nb == 6)
    pct = lambda a: [float(np.percentile(np.where(np.isinf(a), 1e3, a), p)) for p in (5, 50, 95)]
    nomb = nom["broken"]; nomn = nomb.sum(1)
    order = []
    for i in range(19):
        tb = nom["t_break"][i]; idx = np.argsort(np.where(np.isnan(tb), np.inf, tb))
        order.append([int(j + 1) for j in idx if not np.isnan(tb[j])])
    pred = dict(
        written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        blind=True, units=dict(velocity="m/s", displacement="mm", time="ms"),
        model="reduced-order: Timoshenko rod + flange hinge + bolt ring with 6 tension/shear fracture springs + contact/friction; see README",
        Q1_min_dV_any_fastener=dict(nominal=float(dv[nomn >= 1].min()) if (nomn >= 1).any() else None, p5=pct(q1)[0], p50=pct(q1)[1], p95=pct(q1)[2],
                                    fraction_of_samples_with_no_fracture_in_any_test=float(np.isinf(q1).mean())),
        Q2_min_dV_all_fasteners=dict(nominal=float(dv[nomn == 6].min()) if (nomn == 6).any() else None, p5=pct(q2)[0], p50=pct(q2)[1], p95=pct(q2)[2],
                                     fraction_of_samples_never_all_six=float(np.isinf(q2).mean())),
        Q3_per_test=[dict(test=int(S.TESTS[i, 0]), dV_fts=float(S.TESTS[i, 1]), dV_ms=float(dv[i]), peak_g=float(S.TESTS[i, 2]),
                          n_fractured_nominal=int(nomn[i]), fractured_nominal=[int(b) for b in nomb[i]], order_nominal=order[i],
                          n_fractured_median=float(np.median(nb[:, i])), n_p5=float(np.percentile(nb[:, i], 5)), n_p95=float(np.percentile(nb[:, i], 95)),
                          prob_any=float((nb[:, i] >= 1).mean()), prob_all=float((nb[:, i] == 6).mean()),
                          prob_each=[float(v) for v in brk[:, i, :].mean(0)],
                          bolt1_damage_nominal=float(nom["kappa"][i, 0]), flange_hinge_plastic_rot_nominal=float(nom["final"]["hinge_plastic_rot"][i])) for i in range(19)],
        Q3c_text="Fracture starts at fastener 1 (top), which carries the largest tension from the opening of the joint about a heel near the "
                 "bottom of the rod footprint; fasteners 2 and 6 follow within tens of microseconds, then 3 and 5, fastener 4 last (it is at "
                 "the heel and sees mostly shear). Fracture is tension-dominated plastic elongation of the threaded shank; once the top three "
                 "go, the joint unzips in the same impact. Fractures occur 0.4-1.5 ms after the start of the pulse.",
        Q4_block_or_keensert_fracture=dict(answer=0, basis="judgment, not modelled: steel Keenserts in a 4 in aluminium block are stronger than the 10-32 threads"),
        Q5_cantilever_fracture=dict(answer=0, basis="judgment: the reduced model predicts plastic bending of the flange and yielding at the rod root, no fracture model; low confidence"),
    )
    # Q6 / Q7 from the nominal run
    h = nom["hist"]; tt = h["t"]
    q67 = []
    for tno in (2, 4, 7, 9, 11, 13, 16):
        i = tno - 1
        Tp = time_points(nom["pulse"]["A"][i], nom["pulse"]["W"][i], nom["pulse"]["n"][i])
        rows = []
        for T in Tp:
            k = int(np.argmin(np.abs(tt - T)))
            VB, WB, VC, WC = point_at(h, k, i)
            rows.append(dict(t_ms=float(T * 1e3), B=dict(U=0.0, V=VB, W=WB), C=dict(U=0.0, V=VC, W=WC)))
        q67.append(dict(test=tno, n_fractured_nominal=int(nomn[i]), samples=rows))
    pred["Q6_Q7_relative_displacement"] = q67
    pred["selfchecks"] = checks
    pred["hinge_source"] = "results/r3_flange_hinge.json" if hinge else "none (rigid-plastic cap off)"
    json.dump(pred, open(os.path.join(RES, "prediction_draft.json"), "w"), indent=1)
    np.savez(os.path.join(RES, "r3_mc.npz"), nb=nb, brk=brk, q1=q1, q2=q2, dv=dv, **{"P_" + k: v for k, v in P1.items()})
    np.savez(os.path.join(RES, "r3_nominal_hist.npz"), t=h["t"], **{k: v for k, v in h.items() if k != "t"}, broken=nomb, t_break=nom["t_break"], kappa=nom["kappa"])
    print("Q1", pred["Q1_min_dV_any_fastener"]); print("Q2", pred["Q2_min_dV_all_fasteners"])
    for r in pred["Q3_per_test"]:
        print(r["test"], r["dV_fts"], "nom", r["n_fractured_nominal"], r["order_nominal"], "median %.0f [%g-%g] P(any) %.2f P(all) %.2f k1 %.2f hinge %.4f" % (
            r["n_fractured_median"], r["n_p5"], r["n_p95"], r["prob_any"], r["prob_all"], r["bolt1_damage_nominal"], r["flange_hinge_plastic_rot_nominal"]))
    print("seconds", time.time() - t0)
