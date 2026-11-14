"""Sway robustness (reported, not pass/fail): laterally free shuttle with a constant sideways dead load = 2 % of the target's peak force."""
import json, numpy as np, ccx, design, verify
out = {}
for T in ("T1", "T2", "T3"):
    p = np.array(json.load(open(f"results/design_{T}.json"))["p"])
    v = json.load(open(f"results/verify_{T}.json"))
    tgt = np.array(v["target_sigma"])
    fx = 0.02 * tgt.max() / design.KPA
    d, F, info = ccx.run(f"{T}_sway", p, Lh=design.LH, ws=design.WS, dmax=12.5, ninc=100, lateral="free", fx=fx)
    done = info["completed"]
    sig = np.interp(design.DS, d, F) * design.KPA if done else None
    ux = np.array(info.get("ux", []))
    out[T] = dict(fx_N_per_mm=fx, completed=done, max_abs_ux_mm=info.get("max_abs_ux"), d_at_max_ux=float(d[np.argmax(np.abs(ux))]) if len(ux) else None,
                  ux_no_load_mm=None, nrmse_vs_target=design.nrmse(sig, tgt) if done else None,
                  change_vs_fixed=float(np.sqrt(np.mean((sig - np.array(v["ccx_t3"])) ** 2)) / (tgt.max() - tgt.min())) if done else None,
                  d=d.tolist(), ux=ux.tolist(), sigma=None if sig is None else sig.tolist(), d_reached=float(d[-1]) if len(d) else 0.0)
    print(T, {k: out[T][k] for k in ("fx_N_per_mm", "completed", "max_abs_ux_mm", "d_at_max_ux", "nrmse_vs_target", "change_vs_fixed", "d_reached")}, flush=True)
    json.dump(out, open("results/sway_check.json", "w"), indent=1)
