#!/usr/bin/env python3
"""Generate -> run ccx -> parse tip deflection. Returns dict."""
import subprocess, os, re, sys
import mkmesh

RUN_ENV = dict(os.environ)
RUN_ENV["PATH"] = os.path.expanduser("~/.local/bin") + ":" + RUN_ENV.get("PATH","")
RUN_ENV["MAMBA_ROOT_PREFIX"] = os.path.expanduser("~/projects/.micromamba")
CCX = ["micromamba","run","-n","solidmech","ccx"]
HERE = os.path.dirname(os.path.abspath(__file__))

rho, L, P, sig_lim, defl_lim = 7850.0, 1.0, 2000.0, 200e6, 3.0e-3

def nominal_sig_max(b,hr,ht):
    import numpy as np
    x=np.linspace(0,L,4001)[:-1]; h=hr+(ht-hr)*x/L
    return float(np.max(6*P*(L-x)/(b*h**2)))

def mass(b,hr,ht): return rho*b*L*(hr+ht)/2

def evaluate(job, b, hr, ht, nx=40, nz=6):
    inp=os.path.join(HERE,job+".inp")
    tip_ref,ne,nn = mkmesh.gen(inp,b,hr,ht,nx=nx,nz=nz)
    r=subprocess.run(CCX+[job], cwd=HERE, env=RUN_ENV,
                     capture_output=True, text=True)
    dat=os.path.join(HERE,job+".dat")
    uz=None
    with open(dat) as f:
        for line in f:
            t=line.split()
            if len(t)==4 and t[0]==str(tip_ref):
                uz=float(t[3])
    if uz is None:
        raise RuntimeError("no displacement parsed\n"+r.stdout[-1500:])
    defl=abs(uz)
    return dict(b=b,hr=hr,ht=ht, defl=defl, mass=mass(b,hr,ht),
                sig=nominal_sig_max(b,hr,ht), elems=ne, nodes=nn,
                tip_ref=tip_ref, feasible=(defl<=defl_lim and nominal_sig_max(b,hr,ht)<=sig_lim))

if __name__=="__main__":
    job=sys.argv[1]; b,hr,ht=map(float,sys.argv[2:5])
    nx=int(sys.argv[5]) if len(sys.argv)>5 else 40
    nz=int(sys.argv[6]) if len(sys.argv)>6 else 6
    res=evaluate(job,b,hr,ht,nx,nz)
    print(f"b={b*1e3:.3f} hr={hr*1e3:.3f} ht={ht*1e3:.3f} | "
          f"defl={res['defl']*1e3:.4f}mm sig={res['sig']/1e6:.1f}MPa "
          f"mass={res['mass']:.4f}kg feasible={res['feasible']} "
          f"(ne={res['elems']})")
