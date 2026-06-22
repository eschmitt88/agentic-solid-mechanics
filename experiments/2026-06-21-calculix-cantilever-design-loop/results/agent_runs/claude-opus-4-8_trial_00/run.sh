#!/bin/bash
# usage: run.sh b h
set -e
ENV='PATH=$HOME/.local/bin:$PATH MAMBA_ROOT_PREFIX=$HOME/projects/.micromamba'
PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" micromamba run -n solidmech python gen_beam.py "$1" "$2"
PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" micromamba run -n solidmech ccx beam > /dev/null 2>&1
# parse max |U3| (3rd disp component) from displacement block in beam.dat
PATH="$HOME/.local/bin:$PATH" MAMBA_ROOT_PREFIX="$HOME/projects/.micromamba" micromamba run -n solidmech python - "$1" "$2" <<'PY'
import sys
b,h=float(sys.argv[1]),float(sys.argv[2])
maxu=0.0
cap=False
for ln in open("beam.dat"):
    s=ln.strip()
    if s.startswith("displacements"):
        cap=True; continue
    if cap:
        p=s.split()
        if len(p)>=4:
            try:
                u3=float(p[3]); maxu=max(maxu,abs(u3))
            except ValueError:
                pass
        elif s=="":
            pass
stress=6*2000.0*1.0/(b*h*h)/1e6
print(f"RESULT b={b*1000:.2f}mm h={h*1000:.2f}mm  tip_defl={maxu*1000:.4f}mm  stress_nom={stress:.1f}MPa  mass={7850*b*h:.4f}kg")
PY
