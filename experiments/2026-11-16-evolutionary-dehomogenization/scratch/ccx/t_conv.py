"""CalculiX convention tests: CPS6 + SPRING1 + *BUCKLE (Euler column, spring strip)."""
import os, subprocess, sys, numpy as np
W = os.path.dirname(os.path.abspath(__file__))
CCX = os.path.expanduser("~/projects/.micromamba/envs/solidmech/bin/ccx")


def tri6_rect(Lx, Ly, nx, ny):
    xs = np.linspace(0, Lx, 2 * nx + 1); ys = np.linspace(0, Ly, 2 * ny + 1)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.c_[X.ravel(), Y.ravel()]
    nid = lambda i, j: i * (2 * ny + 1) + j
    T = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = nid(2*i, 2*j), nid(2*i+2, 2*j), nid(2*i+2, 2*j+2), nid(2*i, 2*j+2)
            T.append([a, b, c, nid(2*i+1, 2*j), nid(2*i+2, 2*j+1), nid(2*i+1, 2*j+1)])
            T.append([a, c, d, nid(2*i+1, 2*j+1), nid(2*i+1, 2*j+2), nid(2*i, 2*j+1)])
    return P, np.array(T)


def deck(P, T, body, steps):
    s = "*NODE, NSET=NALL\n" + "".join(f"{i+1},{x:.12g},{y:.12g}\n" for i, (x, y) in enumerate(P))
    s += "*ELEMENT, TYPE=CPS6, ELSET=EALL\n" + "".join(f"{e+1}," + ",".join(str(n + 1) for n in t) + "\n" for e, t in enumerate(T))
    s += "*MATERIAL, NAME=M\n*ELASTIC\n1.0, 0.3\n*SOLID SECTION, ELSET=EALL, MATERIAL=M\n1.0\n"
    return s + body + steps


def run(job, s):
    open(os.path.join(W, job + ".inp"), "w").write(s)
    r = subprocess.run([CCX, job], cwd=W, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"))
    return r.stdout


def dat_factors(job):
    txt = open(os.path.join(W, job + ".dat")).read().splitlines()
    out = []
    for i, l in enumerate(txt):
        if "B U C K L I N G   F A C T O R" in l:
            for m in txt[i + 1:]:
                p = m.split()
                if len(p) == 2 and p[0].isdigit():
                    out.append(float(p[1]))
                elif out and not m.strip() == "" and not p[0].isdigit():
                    break
    return out


# 1. Euler column clamped at bottom, axial load at top (4 x 100), Pcr = pi^2 E I / (4 L^2)
P, T = tri6_rect(4.0, 100.0, 8, 100)
bot = np.where(P[:, 1] < 1e-9)[0]; top = np.where(P[:, 1] > 100 - 1e-9)[0]
fix = "*NSET, NSET=BOT\n" + "".join(f"{n+1}\n" for n in bot) + "*BOUNDARY\nBOT,1,2\n"
cl = "*CLOAD\n" + "".join(f"{n+1},2,{-1.0/len(top):.12g}\n" for n in top)
Pcr = np.pi ** 2 * (4 ** 3 / 12) / (4 * 100 ** 2)
out = run("col_b", deck(P, T, fix, "*STEP\n*BUCKLE\n6\n" + cl + "*NODE FILE\nU\n*END STEP\n"))
print("buckle-only factors / Euler:", np.round(np.array(dat_factors("col_b")) / Pcr, 4))
out = run("col_sb", deck(P, T, fix, "*STEP\n*STATIC\n" + cl + "*NODE FILE\nU\n*END STEP\n*STEP\n*BUCKLE\n6\n" + cl + "*NODE FILE\nU\n*END STEP\n"))
print("static+buckle factors / Euler:", np.round(np.array(dat_factors("col_sb")) / Pcr, 4))
# tension: negative factors?
cl2 = cl.replace(",2,-", ",2,")
out = run("col_t", deck(P, T, fix, "*STEP\n*BUCKLE\n4\n" + cl2 + "*NODE FILE\nU\n*END STEP\n"))
print("tension load factors / Euler:", np.round(np.array(dat_factors("col_t")) / Pcr, 4))

# 2. strip 10 x 1 on SPRING1 springs (k_line=1 per mm) in y, x pinned at one node; uniform top load P=1
P, T = tri6_rect(10.0, 1.0, 20, 2)
bot = np.where(P[:, 1] < 1e-9)[0]; bot = bot[np.argsort(P[bot, 0])]
top = np.where(P[:, 1] > 1 - 1e-9)[0]; top = top[np.argsort(P[top, 0])]
# quadratic edges: Simpson weights per edge (L/6, 4L/6, L/6)
def simpson(nodes_sorted, xs):
    w = np.zeros(len(nodes_sorted))
    for k in range(0, len(nodes_sorted) - 2, 2):
        L = xs[k + 2] - xs[k]; w[k] += L / 6; w[k + 1] += 4 * L / 6; w[k + 2] += L / 6
    return w
wb = simpson(bot, P[bot, 0]); wt = simpson(top, P[top, 0])
nn = len(P)
body = ""
for i, (n, w) in enumerate(zip(bot, wb)):
    body += f"*ELEMENT, TYPE=SPRING1, ELSET=SP{i}\n{100000+i},{n+1}\n*SPRING, ELSET=SP{i}\n2\n{1.0*w:.12g}\n"
x0 = bot[len(bot) // 2]
body += f"*BOUNDARY\n{x0+1},1,1\n"
cl = "*CLOAD\n" + "".join(f"{n+1},2,{-w:.12g}\n" for n, w in zip(top, wt))
out = run("strip", deck(P, T, body, "*STEP\n*STATIC\n" + cl + "*NODE FILE\nU\n*NODE PRINT, NSET=NALL\nU\n*END STEP\n"))
sys.path.insert(0, W + "/../..");
import importlib.util
txt = open(os.path.join(W, "strip.dat")).read().splitlines()
U = {}
for l in txt:
    p = l.split()
    if len(p) == 4 and p[0].isdigit():
        U[int(p[0]) - 1] = (float(p[1]), float(p[2]))
uyb = np.array([U[n][1] for n in bot]); uyt = np.array([U[n][1] for n in top])
print("strip: spring force sum", -(wb * uyb).sum(), "uy bot mean", uyb.mean(), "(expect -0.1); uy top", uyt.mean(), "(expect -0.2)")
print("strip compliance F.u", (wt * -uyt).sum())
