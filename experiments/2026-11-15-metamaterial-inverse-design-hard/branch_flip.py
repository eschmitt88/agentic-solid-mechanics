"""Why trial 9's bistability optimiser stalled: at its stalled design the loading path flips branch under a 1e-4 change of
one parameter, and trial 9's continuation flipped at random.  E_back along a line through the stalled design."""
import json
import numpy as np
import design, fem, snap

T9 = "../2026-11-14-metamaterial-inverse-design/results/"
p = np.array(json.load(open(T9 + "opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"])
hs = np.round(np.arange(-8, 9) * 2.5e-5, 9)
new = snap.FoldProblem(log=open("/dev/null", "w"))
leg = snap.GridProblem(S=fem.Solver(nx=40, ny=2, Lh=design.LH, legacy=True), log=open("/dev/null", "w"))
out = dict(component=5, h=hs.tolist(), E_new=[], E_legacy=[], d_u_new=[], nsnap_new=[], d=snap.DS.tolist())
for h in hs:
    q = p.copy(); q[5] += h
    c = new.state_features(q)
    g = leg.evaluate(q)
    out["E_new"].append(float(c["feat"][2])); out["d_u_new"].append(c["d_u"]); out["nsnap_new"].append(c["nsnap"])
    out["E_legacy"].append(float(g["feat"][2]))
    if abs(abs(h) - 1e-4) < 1e-12:
        out["F_plus" if h > 0 else "F_minus"] = c["F"].tolist()
    print(h, out["E_new"][-1], out["E_legacy"][-1], c["nsnap"], flush=True)
json.dump(out, open("results/branch_flip.json", "w"), indent=1)
