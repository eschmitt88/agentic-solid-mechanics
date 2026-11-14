"""Supplementary to rung 0 (run after the pre-registered Q = 6 check): Q = 12 beam, where eqs (2.47)-(2.48) should apply better."""
import json, numpy as np, fem, qiu
from validate_qiu_lib import pts
l, t, h = 100.0, 0.5, 6.0
c = fem.E_PS * t**3 / 12 * h / l**3
ds = np.unique(np.round(np.concatenate([np.arange(0.025, 2.0, 0.025), np.arange(2.0, 10.5, 0.125), np.arange(10.5, 12.4, 0.025)]), 6))
S = fem.Solver(nx=100, ny=2, Lh=l / 2)
F, us, infos = S.path(fem.cosine_params(h, t), ds)
r = pts(ds, F, h)
ref = qiu.approx_points(fem.E_PS, t**3 / 12, h, l)
D, Fs = qiu.series_curve(h / t); ser = pts(D * h, Fs * c, h)
out = dict(case=dict(l=l, t=t, h=h, Q=h / t), jax_100x2=r, qiu_eq_2_47_2_48=ref, qiu_series=ser,
           rel_err_vs_eq_2_47={k: r[k] / ref[k] - 1 for k in ("f_top", "f_bot")},
           abs_err_over_h_vs_eq_2_48={k: (r[k] - ref[k]) / h for k in ("d_top", "d_bot", "d_end", "d_mid")},
           rel_err_vs_series={k: r[k] / ser[k] - 1 for k in ("f_top", "f_bot")},
           n_switched=int(sum(i["switched"] for i in infos)), d=ds.tolist(), F=F.tolist())
json.dump(out, open("results/rung0_qiu_q12_supplementary.json", "w"), indent=1, default=float)
print(json.dumps({k: out[k] for k in out if k not in ("d", "F")}, indent=1, default=float))
