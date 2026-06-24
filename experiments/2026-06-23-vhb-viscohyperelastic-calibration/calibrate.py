"""Inverse problem: calibrate the Ogden + Bergstrom-Boyce parameters (model.py)
to the VHB 4910 cyclic stress-stretch data by gradient descent (JAX autodiff).

Three calibrations:
  fit_all  — calibrate on all 11 curves (headline fit quality);
  study_A  — AMPLITUDE EXTRAPOLATION: train on small/medium amplitudes
             (lambda_max ~2.25, ~4), predict the large ones (~6.25, ~8.9);
  study_B  — HELD-OUT RATE: train on rates 0.01 & 0.05 /s, predict 0.03 /s.

Per-curve fit quality is reported as R^2 and normalized RMSE; predictions are
saved for plotting. Optimizer: Adam (global-ish) then BFGS polish, both on the
unconstrained parameter vector (model.unpack maps to positive physical params).
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.optimize import minimize

import model as M

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
DATA = HERE / ".." / ".." / "raw" / "data" / "vhb4910-hossain2012"
RESULTS = HERE / "results"

RATE = {"rate1": 0.01, "rate2": 0.03, "rate3": 0.05}
AMP = {"Fig3_1": 2.25, "Fig3_2": 4.0, "Fig4_1": 6.25, "Fig4_2": 8.9}
FILES = [f"{a}_{r}" for a in AMP for r in RATE if not (a == "Fig4_2" and r == "rate2")]


def load():
    lam, dt, sig, meta = [], [], [], []
    for name in FILES:
        d = np.loadtxt(DATA / f"{name}.txt")
        lam.append(d[:, 2]); dt.append(d[:, 1]); sig.append(d[:, 5])
        amp, rate = name.rsplit("_", 1)
        meta.append({"name": name, "amp": AMP[amp], "rate": RATE[rate], "ampkey": amp, "ratekey": rate})
    return (jnp.asarray(np.array(lam)), jnp.asarray(np.array(dt)),
            jnp.asarray(np.array(sig)), meta)


LAM, DT, SIG, META = load()
SCALE = jnp.mean(SIG ** 2, axis=1)  # per-curve stress scale for normalization


def preds_for(theta):
    return jax.vmap(lambda l, d: M.predict(l, d, theta))(LAM, DT)


def make_loss(idx):
    idx = jnp.asarray(idx)

    @jax.jit
    def loss(theta):
        P = jax.vmap(lambda l, d: M.predict(l, d, theta))(LAM[idx], DT[idx])
        rel = jnp.mean((P - SIG[idx]) ** 2, axis=1) / SCALE[idx]
        return jnp.mean(rel)
    return loss


def fit(idx, theta0, adam_steps=800, lr=0.05):
    loss = make_loss(idx)
    g = jax.jit(jax.grad(loss))
    # --- Adam ---
    th = theta0
    m = jnp.zeros_like(th); v = jnp.zeros_like(th)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, adam_steps + 1):
        gr = g(th)
        m = b1 * m + (1 - b1) * gr
        v = b2 * v + (1 - b2) * gr ** 2
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        th = th - lr * mh / (jnp.sqrt(vh) + eps)
    # --- BFGS polish ---
    try:
        res = minimize(loss, th, method="BFGS", options={"maxiter": 300})
        if jnp.isfinite(res.fun) and res.fun <= loss(th):
            th = res.x
    except Exception:  # noqa: BLE001
        pass
    return th, float(loss(th))


def metrics(theta, idx):
    P = np.asarray(preds_for(theta))
    out = []
    for i in idx:
        p, s = P[i], np.asarray(SIG[i])
        ss_res = np.sum((p - s) ** 2); ss_tot = np.sum((s - s.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        nrmse = np.sqrt(np.mean((p - s) ** 2)) / (s.max() - s.min())
        out.append({"name": META[i]["name"], "amp": META[i]["amp"], "rate": META[i]["rate"],
                    "r2": round(float(r2), 4), "nrmse": round(float(nrmse), 4)})
    return out


def idxs(pred=lambda m: False):
    tr = [i for i, m in enumerate(META) if not pred(m)]
    te = [i for i, m in enumerate(META) if pred(m)]
    return tr, te


def main():
    RESULTS.mkdir(exist_ok=True)
    # scale-aware init (targets the data magnitude)
    inv = lambda y: float(np.log(np.expm1(y)))
    theta0 = jnp.array([inv(2.0), 2.0, inv(0.3), 3.0, inv(5.0), inv(10.0), inv(1.0)])

    studies = {
        "fit_all": idxs(lambda m: False),
        "study_A_amplitude_extrapolation": idxs(lambda m: m["ampkey"] in ("Fig4_1", "Fig4_2")),
        "study_B_heldout_rate": idxs(lambda m: m["ratekey"] == "rate2"),
    }
    summary = {}
    all_preds = {}
    for name, (tr, te) in studies.items():
        th, lo = fit(tr, theta0)
        tr_m = metrics(th, tr); te_m = metrics(th, te) if te else []
        phys = {k: round(float(v), 5) for k, v in M.unpack(th).items()}
        summary[name] = {
            "train_loss": round(lo, 5), "params": phys,
            "train_files": [m["name"] for m in tr_m],
            "heldout_files": [m["name"] for m in te_m],
            "train_mean_r2": round(float(np.mean([m["r2"] for m in tr_m])), 4),
            "heldout_mean_r2": (round(float(np.mean([m["r2"] for m in te_m])), 4) if te_m else None),
            "per_curve_train": tr_m, "per_curve_heldout": te_m,
        }
        all_preds[name] = np.asarray(preds_for(th))
        ht = summary[name]["heldout_mean_r2"]
        print(f"{name}: train R2={summary[name]['train_mean_r2']} "
              f"heldout R2={ht}  loss={lo:.4f}")
        for m in te_m:
            print(f"    held-out {m['name']:16s} R2={m['r2']:.3f} nRMSE={m['nrmse']:.3f}")

    np.savez(RESULTS / "predictions.npz",
             lam=np.asarray(LAM), sig=np.asarray(SIG),
             names=np.array([m["name"] for m in META]),
             **{f"pred_{k}": v for k, v in all_preds.items()})
    (RESULTS / "calibration.json").write_text(json.dumps(summary, indent=2))
    metrics_top = {
        "fit_all_mean_r2": summary["fit_all"]["train_mean_r2"],
        "amplitude_extrapolation_heldout_r2": summary["study_A_amplitude_extrapolation"]["heldout_mean_r2"],
        "heldout_rate_r2": summary["study_B_heldout_rate"]["heldout_mean_r2"],
        "n_params": M.NPARAM, "n_curves": len(FILES),
    }
    (HERE / "metrics.json").write_text(json.dumps(metrics_top, indent=2))
    print("\nWrote results/calibration.json, predictions.npz, metrics.json")


if __name__ == "__main__":
    main()
