"""Rung 3: CalculiX 3D elastic-plastic pushover of the flange joint (stub of the rod,
flange, contact on the block, bolts as LINEAR spring beds so that the flange mechanism
is isolated).  Gives the moment - relative rotation law of the flange used as the
'flange hinge' of the reduced model."""
import os, sys, json, subprocess, re
import numpy as np, meshio
import ccx, smc_rom as S
import smc_ccx_check as C

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = ccx.WORK
ARM = 2.881          # distance of the centre of gravity of the part from the block face
# true stress - plastic strain for the 6061-T6511 rod, between axial and radial curves of fig. 3-1
PLASTIC = [(48.0e3, 0.0), (51.5e3, 0.01), (56.0e3, 0.04), (60.0e3, 0.08), (64.0e3, 0.15), (70.0e3, 0.5)]


def mesh_stub(zmax=2.0, h_fl=0.09, h_rod=0.22):
    old = S.L_TOT
    S.L_TOT = zmax
    try:
        X, T = C.mesh(h_fl, h_rod)
    finally:
        S.L_TOT = old
    return X, T


def run(dmax=0.12, k_a=1.0e6, k_s=1.5e5, zmax=2.0, tag="", plastic=True, kfac=5.0, ninc=40):
    X, T = mesh_stub(zmax)
    n0 = len(X); e0 = len(T)
    s = "*NODE, NSET=NPART\n" + "".join(f"{i+1},{x:.10g},{y:.10g},{z:.10g}\n" for i, (x, y, z) in enumerate(X))
    ref, rot = n0 + 1, n0 + 2
    s += f"*NODE, NSET=NREF\n{ref},0,0,{ARM}\n{rot},0,0,{ARM}\n"
    s += "*ELEMENT, TYPE=C3D10, ELSET=EPART\n" + "".join(f"{i+1}," + ",".join(str(n + 1) for n in t) + "\n" for i, t in enumerate(T))
    s += f"*MATERIAL, NAME=AL\n*ELASTIC\n{S.E_AL},{S.NU_AL}\n"
    if plastic:
        s += "*PLASTIC\n" + "".join(f"{a},{b}\n" for a, b in PLASTIC)
    s += "*SOLID SECTION, ELSET=EPART, MATERIAL=AL\n"
    s += ccx.nset("NSYM", np.where((np.abs(X[:, 0]) < 1e-7) & (np.abs(X[:, 2] - zmax) > 1e-7))[0])
    cut = np.where(np.abs(X[:, 2] - zmax) < 1e-7)[0]
    s += ccx.nset("NCUT", cut) + f"*RIGID BODY, NSET=NCUT, REF NODE={ref}, ROT NODE={rot}\n"
    nb0 = n0 + 2
    xs = np.linspace(-0.05, 1.65, 18); ys = np.linspace(-1.65, 1.65, 34); zs = np.array([-0.3, 0.0])
    P = np.array([[x, y, z] for z in zs for y in ys for x in xs]); nx, ny = len(xs), len(ys)
    s += "*NODE, NSET=NBLOCK\n" + "".join(f"{nb0+i+1},{p[0]:.8g},{p[1]:.8g},{p[2]:.8g}\n" for i, p in enumerate(P))
    s += "*ELEMENT, TYPE=C3D8, ELSET=EBLOCK\n"; eb = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = nb0 + 1 + j * nx + i; b = a + nx * ny
            eb += 1; s += f"{e0+eb},{a},{a+1},{a+nx+1},{a+nx},{b},{b+1},{b+nx+1},{b+nx}\n"
    s += f"*MATERIAL, NAME=RIG\n*ELASTIC\n{S.E_AL*1e4},0.3\n*SOLID SECTION, ELSET=EBLOCK, MATERIAL=RIG\n"
    s += "*SURFACE, NAME=SMASTER, TYPE=ELEMENT\n" + "".join(f"{e0+k+1},S2\n" for k in range(eb))
    onb = np.abs(X[:, 2]) < 1e-7
    faces = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
    s += "*SURFACE, NAME=SSLAVE, TYPE=ELEMENT\n"
    for e, t in enumerate(T):
        for fi, f in enumerate(faces):
            if onb[t[list(f)]].all(): s += f"{e+1},S{fi+1}\n"
    front = np.abs(X[:, 2] - S.T_FL) < 1e-7; ne = e0 + eb
    for b, (bx, by, share) in enumerate(C.BOLTS):
        r = np.hypot(X[:, 0] - bx, X[:, 1] - by)
        nodes = np.where(front & (r < C.R_WASH + 1e-6))[0]
        s += f"*ELEMENT, TYPE=SPRING1, ELSET=EAX{b}\n" + "".join(f"{ne+i+1},{n+1}\n" for i, n in enumerate(nodes)); ne += len(nodes)
        s += f"*ELEMENT, TYPE=SPRING1, ELSET=ESH{b}\n" + "".join(f"{ne+i+1},{n+1}\n" for i, n in enumerate(nodes)); ne += len(nodes)
        s += f"*SPRING, ELSET=EAX{b}\n3\n{k_a*share/len(nodes):.8e}\n*SPRING, ELSET=ESH{b}\n2\n{k_s*share/len(nodes):.8e}\n"
        s += ccx.nset(f"NB{b}", nodes)
    s += f"*SURFACE INTERACTION, NAME=SI\n*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{kfac*S.E_AL/0.09:.6g}\n*CONTACT PAIR, INTERACTION=SI, TYPE=SURFACE TO SURFACE\nSSLAVE,SMASTER\n"
    s += f"*NSET, NSET=NR1\n{ref}\n*NSET, NSET=NR2\n{rot}\n"
    s += f"*BOUNDARY\nNSYM,1,1\nNBLOCK,1,3\n{ref},1,1\n{rot},2,3\n"
    s += f"*STEP, INC=1000\n*STATIC\n{1.0/ninc},1.0,1e-5,{1.0/ninc}\n*BOUNDARY\n{ref},2,2,{-dmax}\n"
    s += "*NODE PRINT, NSET=NR1, TOTALS=YES, FREQUENCY=1\nU,RF\n*NODE PRINT, NSET=NR2, FREQUENCY=1\nU\n"
    s += "".join(f"*NODE PRINT, NSET=NB{b}, TOTALS=ONLY, FREQUENCY=1\nRF\n" for b in range(4))
    s += "".join(f"*NODE PRINT, NSET=NB{b}, FREQUENCY=1\nU\n" for b in range(4))
    # (an *EL FILE / *NODE FILE card with FREQUENCY=10 here silently made every *NODE PRINT
    #  above print every 10th increment only: in ccx the last FREQUENCY given wins for all)
    s += "*END STEP\n"
    job = "smc_push" + tag
    open(os.path.join(WORK, job + ".inp"), "w").write(s)
    rc, out, sec = ccx.run(job, threads=3, timeout=6000)
    return job, ccx.summarize_log(out), sec, len(X)


def parse(job):
    lines = open(os.path.join(WORK, job + ".dat")).read().splitlines()
    rec = {}
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.search(r"(displacements|forces|total force).*for set (\S+) and time\s+(\S+)", l)
        if m:
            kind, st, t = m.group(1), m.group(2), float(m.group(3))
            j = i + 1; vals = []
            while j < len(lines) and (not lines[j].strip() or re.match(r"^\s*[-0-9]", lines[j])):
                if lines[j].strip():
                    vals.append([float(v) for v in lines[j].split()])
                j += 1
                if j < len(lines) and lines[j].strip() and not re.match(r"^\s*[-0-9]", lines[j]): break
            rec.setdefault(t, {})[(kind, st)] = np.array(vals)
            i = j
        else:
            i += 1
    out = []
    for t in sorted(rec):
        r = rec[t]
        try:
            d = dict(t=t, uy=float(r[("displacements", "NR1")][0][2]), F=float(-2 * r[("total force", "NR1")][0][1]),
                     rot_x=float(r[("displacements", "NR2")][0][1]))
            d["uz_bolt"] = [float(r[("displacements", f"NB{b}")][:, 3].mean()) for b in range(4)]
            out.append(d)
        except KeyError:
            pass
    return out


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    plastic = "elastic" not in tag
    job, info, sec, nn = run(tag=tag, plastic=plastic, dmax=0.12 if plastic else 0.01, ninc=40 if plastic else 2)
    print(info, sec, nn)
    h = parse(job)
    json.dump(dict(info=info, seconds=sec, nodes=nn, hist=h), open(os.path.join(HERE, "results", f"r3_pushover_ccx{tag}.json"), "w"), indent=1)
    for d in h[::max(1, len(h) // 20)]:
        print(d)
