"""Rung 3 self-checks 3a/3b: CalculiX 3D solid half-model of the cantilevered part
(C3D10), block as a clamped stiff plate, flange/block contact, bolts as spring beds
under the washer footprints.  Compared with the reduced model in smc_rom.py."""
import os, sys, json, subprocess
import numpy as np, meshio
import ccx, smc_rom as S
from meshes import GMSH, ENV

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = ccx.WORK
R_HOLE, R_WASH = 0.1105, 0.223
# CalculiX frame: x horizontal, y UP, z along the rod (flange back face at z = 0)
BOLTS = [(0.0, 1.26, 0.5), (1.26 * np.sin(np.pi / 3), 0.63, 1.0), (1.26 * np.sin(np.pi / 3), -0.63, 1.0), (0.0, -1.26, 0.5)]  # x, y, share in half model


def mesh(h_fl=0.07, h_rod=0.22):
    geo = f"""SetFactory("OpenCASCADE");
Cylinder(1) = {{0,0,0, 0,0,{S.T_FL}, {S.D_FL/2}}};
Cylinder(2) = {{0,0,{S.T_FL}, 0,0,{S.L_TOT - S.T_FL}, {S.D_ROD/2}}};
""" + "".join(f"Cylinder({10+i}) = {{{1.26*np.sin(a)},{1.26*np.cos(a)},-0.1, 0,0,0.5, {R_HOLE}}};\n" for i, a in enumerate(np.deg2rad([0, 60, 120, 180, 240, 300]))) + """
BooleanUnion(20) = { Volume{1}; Delete; }{ Volume{2}; Delete; };
BooleanDifference(21) = { Volume{20}; Delete; }{ Volume{10,11,12,13,14,15}; Delete; };
Box(30) = {0,-2,-1, 2,4,8};
BooleanIntersection(31) = { Volume{21}; Delete; }{ Volume{30}; Delete; };
Physical Volume(1) = {31};
""" + f"""Field[1] = Box; Field[1].VIn = {h_fl}; Field[1].VOut = {h_rod}; Field[1].XMin=-2; Field[1].XMax=2; Field[1].YMin=-2; Field[1].YMax=2; Field[1].ZMin=-1; Field[1].ZMax=0.6; Field[1].Thickness=1.0;
Background Field = 1; Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;
Mesh.ElementOrder = 2; Mesh.HighOrderOptimize = 1;
"""
    gp = os.path.join(HERE, "results", "work", "smc_part.geo"); mp = gp.replace(".geo", ".msh")
    open(gp, "w").write(geo)
    r = subprocess.run(GMSH + [gp, "-3", "-format", "msh22", "-o", mp, "-nt", "4"], env=ENV, capture_output=True, text=True)
    assert os.path.exists(mp), r.stdout[-1500:] + r.stderr[-1500:]
    m = meshio.read(mp)
    tets = np.vstack([c.data for c in m.cells if c.type == "tetra10"])
    used = np.unique(tets); remap = -np.ones(len(m.points), int); remap[used] = np.arange(len(used))
    return m.points[used], remap[tets]


def deck_common(X, T):
    s = "*NODE, NSET=NPART\n" + "".join(f"{i+1},{x:.10g},{y:.10g},{z:.10g}\n" for i, (x, y, z) in enumerate(X))
    s += "*ELEMENT, TYPE=C3D10, ELSET=EPART\n" + "".join(f"{i+1}," + ",".join(str(n + 1) for n in t) + "\n" for i, t in enumerate(T))
    s += f"*MATERIAL, NAME=AL\n*ELASTIC\n{S.E_AL},{S.NU_AL}\n*DENSITY\n{S.RHO:.8e}\n*SOLID SECTION, ELSET=EPART, MATERIAL=AL\n"
    s += ccx.nset("NSYM", np.where(np.abs(X[:, 0]) < 1e-7)[0]) + ccx.nset("NBACK", np.where(np.abs(X[:, 2]) < 1e-7)[0])
    tipc = int(np.argmin(np.linalg.norm(X - np.array([0, 0, S.L_TOT]), axis=1)))
    s += ccx.nset("NTIP", [tipc])
    return s, tipc


def frequency(X, T):
    s, _ = deck_common(X, T)
    s += "*BOUNDARY\nNSYM,1,1\nNBACK,1,3\n*STEP\n*FREQUENCY\n3\n*END STEP\n"
    open(os.path.join(WORK, "smc_freq.inp"), "w").write(s)
    rc, out, sec = ccx.run("smc_freq", threads=3)
    f = []
    lines = open(os.path.join(WORK, "smc_freq.dat")).read().splitlines()
    for i, l in enumerate(lines):
        if "E I G E N V A L U E   O U T P U T" in l:
            for q in lines[i + 6:i + 9]:
                p = q.split()
                if len(p) >= 4: f.append(float(p[3]))
    return f, sec


def static(X, T, k_a, k_s, acc_g=1000.0, kfac=20.0):
    s, tipc = deck_common(X, T)
    n0 = len(X); e0 = len(T)
    # stiff clamped plate as the block face
    xs = np.linspace(-0.05, 1.65, 18); ys = np.linspace(-1.65, 1.65, 34); zs = np.array([-0.3, 0.0])
    P = np.array([[x, y, z] for z in zs for y in ys for x in xs]); nx, ny = len(xs), len(ys)
    s += "*NODE, NSET=NBLOCK\n" + "".join(f"{n0+i+1},{p[0]:.8g},{p[1]:.8g},{p[2]:.8g}\n" for i, p in enumerate(P))
    s += "*ELEMENT, TYPE=C3D8, ELSET=EBLOCK\n"
    eb = 0
    for j in range(ny - 1):
        for i in range(nx - 1):
            a = n0 + 1 + j * nx + i; b = a + nx * ny
            eb += 1; s += f"{e0+eb},{a},{a+1},{a+nx+1},{a+nx},{b},{b+1},{b+nx+1},{b+nx}\n"
    s += f"*MATERIAL, NAME=RIG\n*ELASTIC\n{S.E_AL*1e4},0.3\n*SOLID SECTION, ELSET=EBLOCK, MATERIAL=RIG\n"
    s += "*SURFACE, NAME=SMASTER, TYPE=ELEMENT\n" + "".join(f"{e0+k+1},S2\n" for k in range(eb))
    # slave: tet faces lying in z = 0
    onb = np.abs(X[:, 2]) < 1e-7
    faces = [(0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0)]
    s += "*SURFACE, NAME=SSLAVE, TYPE=ELEMENT\n"
    for e, t in enumerate(T):
        for fi, f in enumerate(faces):
            if onb[t[list(f)]].all(): s += f"{e+1},S{fi+1}\n"
    # bolt spring beds on the washer footprints (front face of the flange, z = t)
    front = np.abs(X[:, 2] - S.T_FL) < 1e-7
    spr = ""; sets = []; ne = e0 + eb
    for b, (bx, by, share) in enumerate(BOLTS):
        r = np.hypot(X[:, 0] - bx, X[:, 1] - by)
        nodes = np.where(front & (r < R_WASH + 1e-6))[0]
        sets.append(nodes)
        s += f"*ELEMENT, TYPE=SPRING1, ELSET=EAX{b}\n" + "".join(f"{ne+i+1},{n+1}\n" for i, n in enumerate(nodes)); ne += len(nodes)
        s += f"*ELEMENT, TYPE=SPRING1, ELSET=ESH{b}\n" + "".join(f"{ne+i+1},{n+1}\n" for i, n in enumerate(nodes)); ne += len(nodes)
        s += f"*SPRING, ELSET=EAX{b}\n3\n{k_a*share/len(nodes):.8e}\n*SPRING, ELSET=ESH{b}\n2\n{k_s*share/len(nodes):.8e}\n"
        s += ccx.nset(f"NB{b}", nodes)
    s += f"*SURFACE INTERACTION, NAME=SI\n*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{kfac*S.E_AL/0.07:.6g}\n*CONTACT PAIR, INTERACTION=SI, TYPE=SURFACE TO SURFACE\nSSLAVE,SMASTER\n"
    s += "*BOUNDARY\nNSYM,1,1\nNBLOCK,1,3\n"
    s += f"*STEP, INC=100\n*STATIC\n0.25,1.0\n*DLOAD\nEPART,GRAV,{acc_g*S.G:.8g},0.,-1.,0.\n"
    s += "*NODE PRINT, NSET=NTIP\nU\n" + "".join(f"*NODE PRINT, NSET=NB{b}\nU\n" for b in range(4)) + "*NODE PRINT, NSET=NBLOCK, TOTALS=ONLY\nRF\n*END STEP\n"
    open(os.path.join(WORK, "smc_static.inp"), "w").write(s)
    rc, out, sec = ccx.run("smc_static", threads=3, timeout=3000)
    info = ccx.summarize_log(out)
    # parse displacements of the last increment
    lines = open(os.path.join(WORK, "smc_static.dat")).read().splitlines()
    disp = {}
    last = max(i for i, l in enumerate(lines) if "displacements (vx,vy,vz) for set NTIP" in l)
    cur = None
    for l in lines[last:]:
        if "displacements" in l: cur = l.split("for set")[1].split()[0]; disp[cur] = []
        elif cur and len(l.split()) == 4:
            try: disp[cur].append([float(v) for v in l.split()])
            except ValueError: pass
    res = dict(info=info, seconds=sec, tip_uy=disp["NTIP"][0][2], bolts=[])
    for b, (bx, by, share) in enumerate(BOLTS):
        d = np.array(disp[f"NB{b}"])
        res["bolts"].append(dict(y_up=by, T_full_bolt=float(k_a * d[:, 3].mean()), V_full_bolt=float(-k_s * d[:, 2].mean()), uz_mean=float(d[:, 3].mean()), nodes=len(d)))
    tot = ccx.parse_dat_totals(os.path.join(WORK, "smc_static.dat"))
    res["RF_block"] = tot[-1][2]
    res["mass_check_lbf"] = float(2 * acc_g and 0)
    return res


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    X, T = mesh()
    out = dict(nodes=len(X), elements=len(T))
    f, sec = frequency(X, T); out["ccx_freq_hz"] = f; out["rom_freq_hz"] = S.clamped_frequency().tolist()
    print(out, flush=True)
    out["static_ccx"] = static(X, T, S.NOMINAL["k_a"], S.NOMINAL["k_s"])
    print(json.dumps(out["static_ccx"], indent=1))
    json.dump(out, open(os.path.join(HERE, "results", "r3_selfcheck_ccx.json"), "w"), indent=1)
