"""Rung 2, CalculiX: shallow ironing on the same meshes as the own solver."""
import sys, json, os, re
import numpy as np
import ccx
import rung2_ironing_own as RI
from rung1_ccx import contact_cards

HERE = os.path.dirname(os.path.abspath(__file__))


def run(h=0.1, ctype="SURFACE TO SURFACE", kfac=50.0, stickfac=0.1, dmax1=0.02, dmax2=0.005, tag="", mu=RI.MU, etype="CPE4", timeout=7200, threads=3):
    g = RI.ironing_mesh(h); X, Q, body, el, fa = g["X"], g["Q"], g["body"], g["el"], g["fa"]
    K = kfac * RI.E_SLAB / h
    job = f"iron_{ctype.replace(' ', '')}_h{h}{tag}"
    deck = "*HEADING\nironing\n" + ccx.nodes_block(X)
    deck += ccx.elements_block(Q[body == 0], etype, "ESLAB") + ccx.elements_block(Q[body == 1], etype, "EIND", start=int((body == 0).sum()))
    deck += ccx.surface_block("SSLAVE", el[g["slab_top"]], fa[g["slab_top"]]) + ccx.surface_block("SMASTER", el[g["ind_surf"]], fa[g["ind_surf"]])
    deck += ccx.nset("NBOT", g["bottom"]) + ccx.nset("NTOP", g["top"])
    for name, E in (("MSLAB", RI.E_SLAB), ("MIND", RI.E_IND)):
        mu_s = E / (2 * (1 + RI.NU)); Kb = E / (3 * (1 - 2 * RI.NU))
        deck += f"*MATERIAL, NAME={name}\n*HYPERELASTIC, NEO HOOKE\n{mu_s/2:.8g},{2/Kb:.8g}\n"
    deck += "*SOLID SECTION, ELSET=ESLAB, MATERIAL=MSLAB\n*SOLID SECTION, ELSET=EIND, MATERIAL=MIND\n"
    deck += contact_cards(ctype, K, mu=mu, stick=stickfac * K)
    deck += "*BOUNDARY\nNBOT,1,2\nNTOP,1,1\n"
    outc = "*NODE PRINT, NSET=NTOP, TOTALS=ONLY, FREQUENCY=1\nRF\n*END STEP\n"
    deck += f"*STEP, NLGEOM, INC=100000\n*STATIC\n{dmax1},1.0,1e-7,{dmax1}\n*BOUNDARY\nNTOP,2,2,-1.0\n" + outc
    deck += f"*STEP, NLGEOM, INC=100000\n*STATIC\n{dmax2},1.0,1e-7,{dmax2}\n*BOUNDARY\nNTOP,1,1,10.0\n" + outc
    open(os.path.join(ccx.WORK, job + ".inp"), "w").write(deck)
    rc, out, sec = ccx.run(job, threads=threads, timeout=timeout)
    info = ccx.summarize_log(out)
    res = dict(h=h, ctype=ctype, kfac=kfac, stickfac=stickfac, dmax1=dmax1, dmax2=dmax2, etype=etype, seconds=sec, rc=rc, **info)
    tot = ccx.parse_dat_totals(os.path.join(ccx.WORK, job + ".dat")) if os.path.exists(os.path.join(ccx.WORK, job + ".dat")) else []
    res["hist"] = [dict(t=t, Rx=f[0], Ry=f[1]) for (_, t, f) in tot]
    res["t_reached"] = tot[-1][1] if tot else 0.0
    res["tail"] = out[-500:] if not info["finished"] else ""
    json.dump(res, open(os.path.join(HERE, "results", f"r2_ironing_ccx_{job}.json"), "w"))
    return res


if __name__ == "__main__":
    os.makedirs(ccx.WORK, exist_ok=True)
    kw = dict(a.split("=") for a in sys.argv[1:])
    conv = dict(h=float, kfac=float, stickfac=float, dmax1=float, dmax2=float, tag=str, ctype=lambda s: s.replace("_", " "), mu=float, etype=str, timeout=float, threads=int)
    kw = {k: conv[k](v) for k, v in kw.items()}
    r = run(**kw)
    s = RI.summary(r["hist"]) if r["t_reached"] >= 1.6 else {}
    print({k: v for k, v in r.items() if k not in ("hist",)}, s)
