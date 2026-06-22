#!/usr/bin/env python3
import sys

# Cantilever beam mesh generator -> beam.inp (C3D20R)
# Args: b h  (metres). Mesh divisions fixed below.
b = float(sys.argv[1])   # width along y
h = float(sys.argv[2])   # height along z
L = 1.0
P = 2000.0               # total downward load (-z)

# element divisions (number of C3D20R elements per direction)
nx, ny, nz = 40, 4, 8

# node grid spacing: quadratic elements -> 2*n+1 nodes per direction
Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1
dx, dy, dz = L/(Nx-1), b/(Ny-1), h/(Nz-1)

# C3D20 serendipity: no mid-face / mid-volume nodes. Mark which grid
# points are real nodes. A node exists unless it is a "center" in >=2
# directions simultaneously (those are the missing interior nodes).
def is_odd(i): return i % 2 == 1

nodes = {}     # (i,j,k) -> id
coords = []    # (id,x,y,z)
nid = 0
def gridpoint_exists(i, j, k):
    # count how many indices are odd (i.e. mid-edge positions)
    odd = is_odd(i) + is_odd(j) + is_odd(k)
    # serendipity element keeps corner (0 odd) and mid-edge (1 odd) nodes,
    # drops mid-face (2 odd) and mid-volume (3 odd)
    return odd <= 1

for i in range(Nx):
    for j in range(Ny):
        for k in range(Nz):
            if gridpoint_exists(i, j, k):
                nid += 1
                nodes[(i, j, k)] = nid
                coords.append((nid, i*dx, j*dy, k*dz))

# C3D20 connectivity order (Abaqus/CalculiX):
# 8 corners then 12 mid-edge nodes.
elems = []
eid = 0
for ei in range(nx):
    for ej in range(ny):
        for ek in range(nz):
            i0, j0, k0 = 2*ei, 2*ej, 2*ek
            # corner offsets
            c = [(0,0,0),(2,0,0),(2,2,0),(0,2,0),
                 (0,0,2),(2,0,2),(2,2,2),(0,2,2)]
            # mid-edge offsets (order per Abaqus C3D20)
            m = [(1,0,0),(2,1,0),(1,2,0),(0,1,0),
                 (1,0,2),(2,1,2),(1,2,2),(0,1,2),
                 (0,0,1),(2,0,1),(2,2,1),(0,2,1)]
            conn = []
            for off in c+m:
                key = (i0+off[0], j0+off[1], k0+off[2])
                conn.append(nodes[key])
            eid += 1
            elems.append((eid, conn))

# node sets
fixed = [nodes[(0,j,k)] for j in range(Ny) for k in range(Nz) if (0,j,k) in nodes]
tip   = [nodes[(Nx-1,j,k)] for j in range(Ny) for k in range(Nz) if (Nx-1,j,k) in nodes]
# tip reference node for reading deflection: pick mid y, top? use centroid-ish node at mid
# choose node at x=L, mid-y, mid-z if exists else any tip node
tipmid = None
jm, km = Ny//2, Nz//2
for cand in [(Nx-1,jm,km)]:
    if cand in nodes:
        tipmid = nodes[cand]
if tipmid is None:
    tipmid = tip[0]

pernode = P/len(tip)

with open("beam.inp", "w") as f:
    f.write("*HEADING\n cantilever b=%.5f h=%.5f\n" % (b, h))
    f.write("*NODE\n")
    for (n, x, y, z) in coords:
        f.write("%d, %.8e, %.8e, %.8e\n" % (n, x, y, z))
    f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
    for (e, conn) in elems:
        line = "%d, " % e + ", ".join(str(c) for c in conn[:15])
        f.write(line + ",\n")
        f.write("      " + ", ".join(str(c) for c in conn[15:]) + "\n")
    f.write("*NSET, NSET=NFIX\n")
    f.write(",\n".join(str(n) for n in fixed) + ",\n")
    f.write("*NSET, NSET=NTIP\n")
    f.write(",\n".join(str(n) for n in tip) + ",\n")
    f.write("*NSET, NSET=NMEAS\n%d,\n" % tipmid)
    f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n2.1e11, 0.3\n")
    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write("*STEP\n*STATIC\n")
    f.write("*BOUNDARY\nNFIX, 1, 3, 0.0\n")
    f.write("*CLOAD\n")
    for n in tip:
        f.write("%d, 3, %.8e\n" % (n, -pernode))
    f.write("*NODE PRINT, NSET=NMEAS\nU\n")
    f.write("*NODE FILE, NSET=NTIP\nU\n")
    f.write("*EL PRINT, ELSET=EALL\nS\n")
    f.write("*END STEP\n")

print("nodes=%d elems=%d tipnodes=%d measnode=%d" % (len(coords), len(elems), len(tip), tipmid))
mass = 7850.0*L*b*h
print("mass=%.4f kg  b=%.4f h=%.4f" % (mass, b, h))
