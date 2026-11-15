"""Rung 2a, CalculiX: Cattaneo-Mindlin partial slip, same mesh as the own solver."""
import sys, json, os
import numpy as np
import meshes as M, ccx
import rung2_mindlin_own as R2
from rung1_ccx import contact_cards

HERE = os.path.dirname(os.path.abspath(__file__))
E, NU, R, A_T, MU, P = R2.E, R2.NU, R2.R, R2.A_T, R2.MU, R2.P


def run(level, ctype="SURFACE TO SURFACE", kfac=50.0, stickfac=0.1, tag="", ninc=10):
    g = R2.build_mesh(level); X, Q, off, el, fa, edges = g["X"], g["Q"], g["off"], g["el"], g["fa"], g["edges"]
    K = kfac * E / g["h_c"]
    job = f"mindlin_{ctype.replace(' ', '')}_L{level}{tag}"
    top = np.where((np.abs(X[:, 1] - R) < 1e-9) & (np.arange(len(X)) < off[1]))[0]
    rn = int(top[np.argmin(np.abs(X[top, 0]))]) + 1
    deck = "*HEADING\nmindlin\n" + ccx.nodes_block(X) + ccx.elements_block(Q, "CPE4", "EALL")
    deck += ccx.surface_block("SSLAVE", el[g["sl"]], fa[g["sl"]]) + ccx.surface_block("SMASTER", el[g["ma"]], fa[g["ma"]])
    deck += ccx.nset("NBOT", np.where(np.abs(X[:, 1] + 1.5 * R) < 1e-9)[0])
    deck += "*EQUATION\n" + "".join(f"2\n{n+1},{d},1.0,{rn},{d},-1.0\n" for n in top if n + 1 != rn for d in (1, 2))
    deck += f"*MATERIAL, NAME=M\n*ELASTIC\n{E},{NU}\n*SOLID SECTION, ELSET=EALL, MATERIAL=M\n"
    deck += contact_cards(ctype, K, mu=MU, stick=stickfac * K)
    deck += f"*BOUNDARY\nNBOT,1,2\n{rn},1,1\n"
    out_cards = "*CONTACT FILE\nCSTR\n*NODE PRINT, NSET=NBOT, TOTALS=ONLY\nRF\n*END STEP\n"
    deck += f"*STEP, INC=1000\n*STATIC\n1.0,1.0\n*BOUNDARY\n{rn},2,2,{-0.05*A_T**2/R}\n" + out_cards
    deck += f"*STEP, INC=1000\n*STATIC\n0.125,1.0\n*BOUNDARY, OP=NEW\nNBOT,1,2\n{rn},1,1\n*CLOAD\n{rn},2,{-P}\n" + out_cards
    for qr in R2.QS:
        deck += f"*STEP, INC=1000\n*STATIC\n{1.0/ninc},1.0\n*BOUNDARY, OP=NEW\nNBOT,1,2\n*CLOAD\n{rn},2,{-P}\n{rn},1,{qr*MU*P}\n" + out_cards
    open(os.path.join(ccx.WORK, job + ".inp"), "w").write(deck)
    rc, out, sec = ccx.run(job, timeout=5400)
    info = ccx.summarize_log(out)
    res = dict(ctype=ctype, kfac=kfac, stickfac=stickfac, level=level, ninc=ninc, seconds=sec, **info, loads=[])
    if not info["finished"]:
        res["tail"] = out[-600:]
        return res
    blocks = [b for b in ccx.parse_frd(os.path.join(ccx.WORK, job + ".frd")) if b["name"].startswith("CONTACT")]
    tot = ccx.parse_dat_totals(os.path.join(ccx.WORK, job + ".dat"))
    sn = np.unique(edges[g["sl"]]); x = X[sn, 0]
    # last block of each step: steps 3,4,5 end at total time 3,4,5 (ccx time restarts per step: take by order)
    ends = [max(i for i, b in enumerate(blocks) if abs(b["time"] - k) < 1e-6) for k in (1, 2, 3, 4, 5)]   # ccx reports total time
    prof = dict(x=x)
    for qr, i in zip(R2.QS, ends[2:]):
        b = blocks[i]; idx = np.searchsorted(b["nodes"], sn)
        p = b["values"][idx, 3]; q = np.hypot(b["values"][idx, 4], b["values"][idx, 5])
        r = R2.measure(x, p, q, qr); r["target"] = qr
        res["loads"].append(r); prof[f"p_{qr}"] = p; prof[f"q_{qr}"] = q
    b = blocks[ends[1]]; idx = np.searchsorted(b["nodes"], sn)
    res["q_max_after_normal"] = float(np.hypot(b["values"][idx, 4], b["values"][idx, 5]).max())
    res["RF_bottom_last"] = tot[-1][2][:2]
    np.savez(os.path.join(HERE, "results", f"r2_{job}.npz"), **prof)
    return res


if __name__ == "__main__":
    os.makedirs(ccx.WORK, exist_ok=True)
    name = sys.argv[1]; res = []
    cases = dict(
        graded=[dict(level=3)],
        probe=[dict(level=1)],
        sweep=[dict(level=2), dict(level=2, stickfac=1.0, tag="_s1"), dict(level=2, stickfac=0.01, tag="_s0.01"),
               dict(level=2, kfac=5.0, tag="_k5"), dict(level=2, kfac=500.0, tag="_k500"),
               dict(level=2, ctype="NODE TO SURFACE", tag=""), dict(level=2, ninc=40, tag="_inc40")])[name]
    for c in cases:
        try:
            r = run(**c)
        except Exception as e:
            r = dict(case=str(c), exception=repr(e))
        res.append(r); print(json.dumps(r)[:1400], flush=True)
        json.dump(res, open(os.path.join(HERE, "results", f"r2_mindlin_ccx_{name}.json"), "w"), indent=1)
