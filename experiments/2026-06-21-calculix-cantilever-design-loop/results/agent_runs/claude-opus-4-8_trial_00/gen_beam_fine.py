import sys, numpy as np

b = float(sys.argv[1])   # width along y (m)
h = float(sys.argv[2])   # height along z (m)
L = 1.0
P = 2000.0               # total downward (-z)

# element divisions
nx, ny, nz = 40, 3, 10    # C3D20R quadratic elements
# node grid: 2*n+1 nodes per element direction
Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1
xs = np.linspace(0, L, Nx)
ys = np.linspace(0, b, Ny)
zs = np.linspace(0, h, Nz)

# C3D20: serendipity - exclude mid-volume / mid-face center nodes.
# Build full grid node ids but only keep nodes used by serendipity elements.
def nid(i, j, k):
    return i*Ny*Nz + j*Nz + k + 1

# C3D20 connectivity (Abaqus/CalculiX order) for an element with corner index (ei,ej,ek)
# corner local positions in (i,j,k) offsets of 0/1/2
def conn(ei, ej, ek):
    i0, j0, k0 = 2*ei, 2*ej, 2*ek
    c = lambda di,dj,dk: nid(i0+di, j0+dj, k0+dk)
    # 8 corners
    n1=c(0,0,0); n2=c(2,0,0); n3=c(2,2,0); n4=c(0,2,0)
    n5=c(0,0,2); n6=c(2,0,2); n7=c(2,2,2); n8=c(0,2,2)
    # 12 mid-edge
    n9 =c(1,0,0); n10=c(2,1,0); n11=c(1,2,0); n12=c(0,1,0)
    n13=c(1,0,2); n14=c(2,1,2); n15=c(1,2,2); n16=c(0,1,2)
    n17=c(0,0,1); n18=c(2,0,1); n19=c(2,2,1); n20=c(0,2,1)
    return [n1,n2,n3,n4,n5,n6,n7,n8,n9,n10,n11,n12,n13,n14,n15,n16,n17,n18,n19,n20]

elems = []
used = set()
eid = 0
for ei in range(nx):
    for ej in range(ny):
        for ek in range(nz):
            eid += 1
            cn = conn(ei, ej, ek)
            elems.append((eid, cn))
            used.update(cn)

# emit nodes (only used ones)
lines = ["*NODE\n"]
coords = {}
for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            n = nid(i,j,k)
            if n in used:
                coords[n] = (xs[i], ys[j], zs[k])
for n in sorted(coords):
    x,y,z = coords[n]
    lines.append(f"{n}, {x:.8e}, {y:.8e}, {z:.8e}\n")

lines.append("*ELEMENT, TYPE=C3D20R, ELSET=BEAM\n")
for eid, cn in elems:
    first = [str(eid)] + [str(x) for x in cn[:15]]   # 16 entries
    rest  = [str(x) for x in cn[15:]]                 # remaining 5
    lines.append(", ".join(first) + ",\n")
    lines.append(", ".join(rest) + "\n")

# node sets: fixed face x=0, tip face x=L
fixed = [n for n in coords if abs(coords[n][0]-0.0) < 1e-9]
tip   = [n for n in coords if abs(coords[n][0]-L)   < 1e-9]

def nset(name, ns):
    out = [f"*NSET, NSET={name}\n"]
    for i in range(0, len(ns), 8):
        out.append(", ".join(str(x) for x in ns[i:i+8]) + ",\n")
    return out

lines += nset("FIX", sorted(fixed))
lines += nset("TIP", sorted(tip))

# distribute total load P over tip nodes via CLOAD (-z = DOF 3)
load_per = -P/len(tip)
lines.append("*BOUNDARY\nFIX, 1, 3\n")
lines.append("*MATERIAL, NAME=STEEL\n*ELASTIC\n2.1e11, 0.3\n")
lines.append("*SOLID SECTION, ELSET=BEAM, MATERIAL=STEEL\n")
lines.append("*STEP\n*STATIC\n")
lines.append("*CLOAD\n")
for n in sorted(tip):
    lines.append(f"{n}, 3, {load_per:.8e}\n")
lines.append("*NODE PRINT, NSET=TIP\nU\n")
lines.append("*END STEP\n")

with open("beam.inp","w") as f:
    f.writelines(lines)
print(f"b={b} h={h} tip_nodes={len(tip)} mass={7850*L*b*h:.4f}")
