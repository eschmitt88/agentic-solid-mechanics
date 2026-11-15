"""Rung 3: reduced-order dynamic model of the Sandia Mechanics Challenge 2023
drop test (cantilevered rod bolted to a block by six 10-32 fasteners).

Units: inch, lbf, second (mass in lbf s^2/in).  Planar motion in the vertical
symmetry plane.  y is positive DOWN (packet convention), s is the distance from
the block face along the rod (packet -Z).

  rod      Timoshenko beam, lumped mass, elastic
  flange   rigid (node 0 of the beam, at the block face)
  bolts    six coupled tension/shear elastic-plastic springs with fracture,
           hole clearance, preload
  contact  flange face on a rigid block: compression-only bed + one Coulomb slider
  load     base acceleration pulse (body force in the block frame)

All arrays carry a leading batch axis so that tests x Monte Carlo samples are
integrated together (explicit central differences).
"""
import numpy as np

G = 386.089
# geometry from drawing A-3 of the packet
L_TOT, D_ROD, D_FL, T_FL, R_BC = 6.250, 1.675, 3.000, 0.250, 1.260
RHO = 0.0975 / G            # 6061, lbf s^2/in^4
E_AL, NU_AL = 10.0e6, 0.33
# fastener 1 at the top, numbering clockwise seen from the front; y positive down
BOLT_ANG = np.deg2rad([0, 60, 120, 180, 240, 300])
BOLT_Y = -R_BC * np.cos(BOLT_ANG)
BOLT_X = R_BC * np.sin(BOLT_ANG)

# Table 4-1 of the packet: test, dV (ft/s), peak (g), duration at 10 % (ms)
TESTS = np.array([
    [1, 43.2, 5151, 0.46], [2, 50.0, 5995, 0.45], [3, 54.2, 6312, 0.46], [4, 60.0, 7388, 0.44],
    [5, 63.7, 7449, 0.44], [6, 66.4, 9209, 0.33], [7, 70.8, 8634, 0.43], [8, 77.9, 8562, 0.49],
    [9, 82.6, 9567, 0.46], [10, 85.2, 10030, 0.45], [11, 89.1, 10858, 0.43], [12, 95.9, 10670, 0.46],
    [13, 98.8, 11924, 0.43], [14, 103.9, 11692, 0.45], [15, 105.6, 11860, 0.45], [16, 109.1, 11578, 0.49],
    [17, 114.9, 10080, 0.64], [18, 119.7, 11114, 0.65], [19, 127.9, 8774, 0.88]])

# fastener curves read off figures 3-5 .. 3-7 of the packet (see config.yaml for uncertainty)
# tension: force (lbf) against kappa = plastic elongation / plastic elongation at fracture
T_KAPPA = np.array([0.0, 0.10, 0.50, 0.80, 0.95, 1.0])
T_FORCE = np.array([4000., 4200., 4100., 3600., 3200., 3000.])
# single shear plane = half of the double-shear force
V_KAPPA = np.array([0.0, 0.30, 0.70, 1.0])
V_FORCE = np.array([1300., 1800., 2100., 2150.])

NOMINAL = dict(fT=1.0, d_tf=0.036, fV=1.0, d_sf=0.018, k_a=1.0e6, k_s=1.5e5, F_pre=1600.0, mu=0.4,
               clear=0.0155, k_c=2.0e7, fA=1.0, k_t=5.0e6,
               # flange hinge between the rod root and the bolt ring (added after self-check 3b missed):
               k_f=2.5e6, r_c=1.10, fM=1.0)   # calibrated on results/r3_selfcheck_ccx.json (tip +0.1 %, T1 +1.2 %)
# flange hinge: yield moment (in-lbf) against accumulated plastic rotation (rad); overwritten
# from the CalculiX pushover (results/r3_flange_hinge.json) when that file exists
HINGE_TH = np.array([0.0, 1.0])
HINGE_M = np.array([1.0e30, 1.0e30])


def pulse_params(dv_fts, peak_g, d10_ms):
    """pulse a(t) = A * hav(t/W)^n.  Peak and velocity change are always reproduced
    exactly (W = dV / (A * integral)); the exponent n is chosen so that the duration at
    10 % of peak matches the table as well, limited to 0.5 <= n <= 4 (n = 1 is the
    haversine the packet names).  Where the limit binds, the 10 % duration is off."""
    A = peak_g * G; dv = dv_fts * 12.0; d10 = d10_ms * 1e-3
    x = np.linspace(0, 1, 20001)
    hav = 0.5 * (1 - np.cos(2 * np.pi * x))
    def props(n):
        s = hav ** n
        on = x[s >= 0.1]
        return np.trapezoid(s, x), (on.max() - on.min())
    def d10_of(n):
        integ, wid = props(n)
        return dv / (A * integ) * wid
    lo, hi = 0.5, 4.0                    # d10_of(n) increases with n
    if d10_of(hi) < d10: n = hi          # table pulse more peaked than n = 4 allows
    elif d10_of(lo) > d10: n = lo        # table pulse fatter than n = 0.5 allows
    else:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if d10_of(mid) < d10: lo = mid
            else: hi = mid
        n = 0.5 * (lo + hi)
    W = dv / (A * props(n)[0])
    return A, W, n


def pulse_d10(dv_fts, peak_g, d10_ms):
    A, W, n = pulse_params(dv_fts, peak_g, d10_ms)
    x = np.linspace(0, 1, 20001); s = (0.5 * (1 - np.cos(2 * np.pi * x))) ** n
    on = x[s >= 0.1]
    return (on.max() - on.min()) * W * 1e3


def beam_matrices(nel=13):
    n = nel + 1; Le = L_TOT / nel
    A = np.pi * D_ROD ** 2 / 4; I = np.pi * D_ROD ** 4 / 64
    Gs = E_AL / (2 * (1 + NU_AL)); ks = 6 * (1 + NU_AL) / (7 + 6 * NU_AL)
    phi = 12 * E_AL * I / (ks * Gs * A * Le ** 2)
    c = E_AL * I / ((1 + phi) * Le ** 3)
    ke = c * np.array([[12, 6 * Le, -12, 6 * Le], [6 * Le, (4 + phi) * Le ** 2, -6 * Le, (2 - phi) * Le ** 2],
                       [-12, -6 * Le, 12, -6 * Le], [6 * Le, (2 - phi) * Le ** 2, -6 * Le, (4 + phi) * Le ** 2]])
    Kb = np.zeros((2 * n, 2 * n)); Ka = np.zeros((n, n))
    for e in range(nel):
        idx = [2 * e, 2 * e + 1, 2 * e + 2, 2 * e + 3]
        Kb[np.ix_(idx, idx)] += ke
        Ka[np.ix_([e, e + 1], [e, e + 1])] += E_AL * A / Le * np.array([[1, -1], [-1, 1]])
    m = np.full(n, RHO * A * Le); m[0] *= 0.5; m[-1] *= 0.5
    J = np.full(n, RHO * I * Le); J[0] *= 0.5; J[-1] *= 0.5
    # flange material outside the rod radius, lumped on node 0
    A_ann = np.pi * (D_FL ** 2 - D_ROD ** 2) / 4
    m[0] += RHO * A_ann * T_FL
    J[0] += RHO * (np.pi * (D_FL ** 4 - D_ROD ** 4) / 64) * T_FL
    return dict(n=n, Le=Le, Kb=Kb, Ka=Ka, m=m, J=J, A=A, I=I)


def face_points(nr=10, nt=24):
    r = (np.arange(nr) + 0.5) / nr * (D_FL / 2)
    th = (np.arange(nt) + 0.5) / nt * 2 * np.pi
    R, T = np.meshgrid(r, th, indexing="ij")
    area = (D_FL / 2 / nr) * (2 * np.pi * R / nt)
    return (R * np.sin(T)).ravel(), (-R * np.cos(T)).ravel(), area.ravel()


def simulate(tests, P, dt=2.5e-7, t_end=2.6e-3, nel=13, sample_dt=1e-5, linear=False, static_acc=None, damping=2e-7):
    """tests: (B,3) rows (dV ft/s, peak g, d10 ms); P: dict of parameter arrays (B,) and 'clear' (B,6).
    linear=True: no plasticity, no fracture.  static_acc: if given (B,), a slowly ramped constant
    acceleration (g) with heavy damping instead of the pulse (static check)."""
    B = len(tests); bm = beam_matrices(nel); n = bm["n"]
    Kb, Ka, m, J = bm["Kb"], bm["Ka"], bm["m"], bm["J"]
    xu, yu, au = face_points()
    sc_r = P["r_c"] / (D_FL / 2)                      # effective contact radius (calibrated)
    yk = yu[None, :] * sc_r[:, None]; ak = au[None, :] * sc_r[:, None] ** 2
    Aface = ak.sum(1)
    k_f = P["k_f"]; rigid_hinge = k_f > 1e20
    Jf = RHO * (np.pi * (D_FL ** 4 - D_ROD ** 4) / 64) * T_FL       # rotary inertia of the flange ring
    phif = np.zeros(B); wf = np.zeros(B); thp = np.zeros(B); thacc = np.zeros(B)
    pp = np.array([pulse_params(*t) for t in tests])
    A, W, npow = pp[:, 0] * P["fA"], pp[:, 1], pp[:, 2]
    k_a, k_s, k_c, mu, k_t = P["k_a"], P["k_s"], P["k_c"], P["mu"], P["k_t"]
    clear = P["clear"]; clear_nom = P.get("clear_nom", np.full(B, 0.0155))
    # preloaded initial state
    w_init = -6 * P["F_pre"] / (k_c * Aface)
    e_pre = P["F_pre"] / k_a - w_init
    ub = np.zeros((B, 2 * n)); vb = np.zeros((B, 2 * n))      # (v, phi) per node
    ua = np.repeat(w_init[:, None], n, axis=1); va = np.zeros((B, n))
    ep = np.zeros((B, 6)); sp = np.zeros((B, 6)); kap = np.zeros((B, 6)); broken = np.zeros((B, 6), bool)
    t_break = np.full((B, 6), np.nan)
    v_sl = np.zeros(B)
    mb = np.empty(2 * n); mb[0::2] = m; mb[1::2] = J
    nst = int(round(t_end / dt)); ns = int(round(sample_dt / dt))
    hist = dict(t=[], v0=[], w0=[], phi0=[], phif=[], vt=[], wt=[], phit=[], T=[], V=[], N=[], Ff=[], Mh=[])
    en = dict(Wext=np.zeros(B), Wpl=np.zeros(B), Wfr=np.zeros(B), Wdamp=np.zeros(B), Wfrac=np.zeros(B), Wh=np.zeros(B))
    peak = dict(T=np.zeros((B, 6)), V=np.zeros((B, 6)), ep=np.zeros((B, 6)), sp=np.zeros((B, 6)), root_M=np.zeros(B))

    def forces(t, ub, ua, vb, va, update):
        nonlocal ep, sp, kap, broken, v_sl, t_break, thp, thacc
        v0, w0 = ub[:, 0], ua[:, 0]
        phir = ub[:, 1]                                   # rotation of the rod root
        phi0 = np.where(rigid_hinge, phir, phif)          # rotation of the bolt ring
        # --- flange hinge (elastic-plastic rotational spring between rod root and bolt ring)
        Mtr = np.where(rigid_hinge, 0.0, k_f * (phir - phif - thp))
        My = P["fM"] * np.interp(thacc, HINGE_TH, HINGE_M)
        Mh = np.clip(Mtr, -My, My)
        dth = np.where(rigid_hinge, 0.0, (Mtr - Mh) / k_f)
        if update:
            thp = thp + dth; thacc = thacc + np.abs(dth); en["Wh"] += np.abs(Mh * dth)
        # --- contact bed
        g = w0[:, None] - phi0[:, None] * yk
        pk = k_c[:, None] * np.maximum(0.0, -g) * ak
        N = pk.sum(1); Mc = -(pk * yk).sum(1)
        # --- friction slider
        Ff = k_t * (v0 - v_sl); lim = mu * N
        over = np.abs(Ff) > lim
        Ff = np.where(over, lim * np.sign(Ff), Ff)
        vsl_new = np.where(over, v0 - Ff / k_t, v_sl)
        # --- bolts
        e = w0[:, None] - phi0[:, None] * BOLT_Y[None, :]
        Ttr = np.maximum(0.0, k_a[:, None] * (e + e_pre[:, None] - ep))
        veff = v0[:, None] - sp
        back = 2 * clear_nom[:, None] - clear
        d = np.where(veff > clear, veff - clear, np.where(veff < -back, veff + back, 0.0))
        Vtr = k_s[:, None] * d
        if linear:
            T, V = Ttr, Vtr; dep = dsp = np.zeros_like(T); kap_new = kap
        else:
            Ty = P["fT"][:, None] * np.interp(kap, T_KAPPA, T_FORCE); Vy = P["fV"][:, None] * np.interp(kap, V_KAPPA, V_FORCE)
            f = np.sqrt((Ttr / Ty) ** 2 + (Vtr / Vy) ** 2)
            sc = np.where(f > 1.0, 1.0 / np.maximum(f, 1e-30), 1.0)
            T, V = Ttr * sc, Vtr * sc
            dep = (Ttr - T) / k_a[:, None]; dsp = (Vtr - V) / k_s[:, None]
            kap_new = kap + np.sqrt((dep / P["d_tf"][:, None]) ** 2 + (dsp / P["d_sf"][:, None]) ** 2)
        T = np.where(broken, 0.0, T); V = np.where(broken, 0.0, V)
        if update:
            if not linear:
                en["Wpl"] += np.sum(np.where(broken, 0.0, T * dep + np.abs(V * dsp)), axis=1)
                ep = np.where(broken, ep, ep + dep); sp = np.where(broken, sp, sp + dsp); kap = np.where(broken, kap, kap_new)
                newly = (~broken) & (kap >= 1.0)
                # elastic energy stored in a fastener is lost when it breaks
                en["Wfrac"] += np.sum(np.where(newly, 0.5 * T ** 2 / k_a[:, None] + 0.5 * V ** 2 / k_s[:, None], 0.0), axis=1)
                t_break = np.where(newly, t, t_break); broken = broken | newly
            en["Wfr"] += np.abs(Ff * (vsl_new - v_sl)); v_sl = vsl_new
            peak["T"] = np.maximum(peak["T"], T); peak["V"] = np.maximum(peak["V"], np.abs(V))
        Fb = -ub @ Kb.T - damping * (vb @ Kb.T)
        Fa = -ua @ Ka.T - damping * (va @ Ka.T)
        Mjoint = (T * BOLT_Y[None, :]).sum(1) + Mc
        Fb[:, 0] += -V.sum(1) - Ff
        Fb[:, 1] += np.where(rigid_hinge, Mjoint, -Mh)
        Fa[:, 0] += -T.sum(1) + N
        self_Mf = Mjoint + Mh                              # moment on the bolt ring
        return Fb, Fa, T, V, N, Ff, self_Mf, Mh

    def acc(t):
        if static_acc is not None:
            return static_acc * G * min(1.0, t / (0.6 * t_end))
        x = np.clip(t / W, 0, 1)
        return A * (0.5 * (1 - np.cos(2 * np.pi * x))) ** npow

    cdamp = 0.0 if static_acc is None else 4.0e4      # mass-proportional damping for the static ramp
    Fb, Fa, T, V, N, Ff, Mf, Mh = forces(0.0, ub, ua, vb, va, False)
    ab = np.zeros_like(ub); aa = np.zeros_like(ua); af = np.zeros(B); vf = np.zeros(B)
    for i in range(nst):
        t = i * dt
        vb += 0.5 * dt * ab; va += 0.5 * dt * aa; vf += 0.5 * dt * af
        ub += dt * vb; ua += dt * va; phif += dt * vf
        Fb, Fa, T, V, N, Ff, Mf, Mh = forces(t + dt, ub, ua, vb, va, True)
        a_t = acc(t + dt)
        fext = a_t[:, None] * m[None, :]
        Fb[:, 0::2] += fext
        ab = Fb / mb[None, :] - cdamp * vb; aa = Fa / m[None, :] - cdamp * va
        af = np.where(rigid_hinge, 0.0, Mf / Jf - (cdamp + 2.0e4) * vf)   # ring: light extra damping (its own mode is ~30 kHz)
        vb += 0.5 * dt * ab; va += 0.5 * dt * aa; vf += 0.5 * dt * af
        en["Wext"] += np.sum(fext * vb[:, 0::2], axis=1) * dt
        en["Wdamp"] += damping * dt * (np.einsum("bi,ij,bj->b", vb, Kb, vb) + np.einsum("bi,ij,bj->b", va, Ka, va))
        en["Wdamp"] += np.where(rigid_hinge, 0.0, Jf * 2.0e4 * vf ** 2 * dt)
        peak["ep"] = np.maximum(peak["ep"], ep); peak["sp"] = np.maximum(peak["sp"], np.abs(sp))
        if i % ns == 0:
            hist["t"].append(t + dt); hist["v0"].append(ub[:, 0].copy()); hist["phi0"].append(ub[:, 1].copy()); hist["phif"].append(np.where(rigid_hinge, ub[:, 1], phif)); hist["Mh"].append(Mh.copy()); hist["w0"].append(ua[:, 0] - w_init)
            hist["vt"].append(ub[:, -2].copy()); hist["phit"].append(ub[:, -1].copy()); hist["wt"].append(ua[:, -1] - w_init)
            hist["T"].append(T.copy()); hist["V"].append(V.copy()); hist["N"].append(N.copy()); hist["Ff"].append(Ff.copy())
            peak["root_M"] = np.maximum(peak["root_M"], np.abs(ub @ Kb[1]))
    KE = 0.5 * np.sum(mb[None, :] * vb ** 2, axis=1) + 0.5 * np.sum(m[None, :] * va ** 2, axis=1) + np.where(rigid_hinge, 0.0, 0.5 * Jf * vf ** 2)
    ua0 = np.repeat(w_init[:, None], n, axis=1)
    # elastic energy relative to the preloaded start (bolts, bed, slider, beam)
    def elastic(ub_, ua_, ep_, sp_, vsl_, brk, phif_, thp_):
        pr = np.where(rigid_hinge, ub_[:, 1], phif_)[:, None]
        g = ua_[:, :1] - pr * yk
        Ec = 0.5 * np.sum(k_c[:, None] * np.maximum(0, -g) ** 2 * ak, axis=1)
        Ec = Ec + np.where(rigid_hinge, 0.0, 0.5 * np.where(rigid_hinge, 0.0, k_f) * (ub_[:, 1] - phif_ - thp_) ** 2)
        e = ua_[:, :1] - pr * BOLT_Y[None, :]
        Tt = np.maximum(0, k_a[:, None] * (e + e_pre[:, None] - ep_))
        veff = ub_[:, :1] - sp_; back = 2 * clear_nom[:, None] - clear
        d = np.where(veff > clear, veff - clear, np.where(veff < -back, veff + back, 0.0))
        Eb = np.sum(np.where(brk, 0, 0.5 * Tt ** 2 / k_a[:, None] + 0.5 * k_s[:, None] * d ** 2), axis=1)
        Es = 0.5 * k_t * (ub_[:, 0] - vsl_) ** 2
        Ebm = 0.5 * np.einsum("bi,ij,bj->b", ub_, Kb, ub_) + 0.5 * np.einsum("bi,ij,bj->b", ua_, Ka, ua_)
        return Ec + Eb + Es + Ebm
    E0 = elastic(np.zeros_like(ub), ua0, np.zeros_like(ep), np.zeros_like(sp), np.zeros(B), np.zeros_like(broken), np.zeros(B), np.zeros(B))
    E1 = elastic(ub, ua, ep, sp, v_sl, broken, phif, thp)
    out = dict(broken=broken, t_break=t_break, peak=peak, kappa=kap, ep=ep, sp=sp, pulse=dict(A=A, W=W, n=npow),
               energy=dict(Wext=en["Wext"], KE=KE, dE=E1 - E0, Wpl=en["Wpl"], Wfr=en["Wfr"], Wdamp=en["Wdamp"], Wfrac=en["Wfrac"], Wh=en["Wh"]),
               hist={k: np.array(v) for k, v in hist.items()}, final=dict(v0=ub[:, 0], phi0=ub[:, 1], w0=ua[:, 0] - w_init, vt=ub[:, -2], hinge_plastic_rot=thp))
    return out


def params(B, rng=None, **over):
    """nominal parameters (rng None) or Monte Carlo draws."""
    P = {k: np.full(B, v, float) for k, v in NOMINAL.items() if k != "clear"}
    P["clear"] = np.full((B, 6), NOMINAL["clear"]); P["clear_nom"] = np.full(B, NOMINAL["clear"])
    if rng is not None:
        U = lambda a, b, s=B: rng.uniform(a, b, s)
        LU = lambda a, b: np.exp(rng.uniform(np.log(a), np.log(b), B))
        P.update(fT=U(0.90, 1.15), d_tf=U(0.030, 0.040), fV=U(0.90, 1.10), d_sf=U(0.014, 0.022),
                 k_a=LU(0.6, 1.6) * NOMINAL["k_a"], k_s=LU(1.0e5, 3.0e5), F_pre=U(1200, 2200), mu=U(0.2, 0.6),
                 k_c=LU(1e7, 4e7), fA=U(0.95, 1.05))
        P["clear"] = U(0.0, 0.031, (B, 6))
    for k, v in over.items():
        P[k] = np.broadcast_to(np.asarray(v, float), P[k].shape).copy()
    return P


def clamped_frequency(nel=13):
    bm = beam_matrices(nel); n = bm["n"]
    mb = np.empty(2 * n); mb[0::2] = bm["m"]; mb[1::2] = bm["J"]
    K = bm["Kb"][2:, 2:]; Mi = 1 / np.sqrt(mb[2:])
    w2 = np.linalg.eigvalsh(K * Mi[:, None] * Mi[None, :])
    return np.sqrt(w2[:3]) / (2 * np.pi)
