import sys

L, b, h = 1.0, 0.05, 0.1
nx, ny, nz = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

# Grid index ranges: i in 0..2nx (x), j in 0..2ny (y), k in 0..2nz (z)
Nx, Ny, Nz = 2*nx+1, 2*ny+1, 2*nz+1

def valid(i, j, k):
    return (i % 2) + (j % 2) + (k % 2) <= 1

# assign node numbers
node_id = {}
nodes = []
nid = 0
for k in range(Nz):
    for j in range(Ny):
        for i in range(Nx):
            if valid(i, j, k):
                nid += 1
                node_id[(i, j, k)] = nid
                x = L * i / (2*nx)
                y = b * j / (2*ny)
                z = h * k / (2*nz)
                nodes.append((nid, x, y, z))

# C3D20 local node offsets (i,j,k) within element, in Abaqus order 1..20
offs = [
    (0,0,0),(2,0,0),(2,2,0),(0,2,0),
    (0,0,2),(2,0,2),(2,2,2),(0,2,2),
    (1,0,0),(2,1,0),(1,2,0),(0,1,0),
    (1,0,2),(2,1,2),(1,2,2),(0,1,2),
    (0,0,1),(2,0,1),(2,2,1),(0,2,1),
]

elems = []
eid = 0
for ez in range(nz):
    for ey in range(ny):
        for ex in range(nx):
            bi, bj, bk = 2*ex, 2*ey, 2*ez
            conn = [node_id[(bi+di, bj+dj, bk+dk)] for (di,dj,dk) in offs]
            eid += 1
            elems.append((eid, conn))

# node sets
fixed = [n for (i,j,k), n in node_id.items() if i == 0]  # x=0
tip   = [n for (i,j,k), n in node_id.items() if i == 2*nx]  # x=L

with open("beam.inp", "w") as f:
    f.write("*HEADING\nCantilever beam C3D20R\n")
    f.write("*NODE\n")
    for nid, x, y, z in nodes:
        f.write(f"{nid}, {x:.10g}, {y:.10g}, {z:.10g}\n")
    f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
    for eid, conn in elems:
        vals = [eid] + conn  # 21 fields total
        # max 16 fields per line; first line 16 (eid+15 nodes), continuation 5
        f.write(", ".join(str(v) for v in vals[:16]) + ",\n")
        f.write(", ".join(str(v) for v in vals[16:]) + "\n")
    f.write("*NSET, NSET=FIX\n")
    for n in sorted(fixed):
        f.write(f"{n},\n")
    f.write("*NSET, NSET=TIP\n")
    for n in sorted(tip):
        f.write(f"{n},\n")
    f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n2.1e11, 0.3\n")
    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write("*STEP\n*STATIC\n")
    f.write("*BOUNDARY\nFIX, 1, 3\n")
    # total load P=1000 N in -z distributed over tip nodes
    P = 1000.0
    per = P / len(tip)
    f.write("*CLOAD\n")
    for n in sorted(tip):
        f.write(f"{n}, 3, {-per:.10g}\n")
    f.write("*NODE PRINT, NSET=TIP\nU\n")
    f.write("*EL PRINT, ELSET=EALL\nS\n")
    f.write("*NODE FILE\nU\n*EL FILE\nS\n")
    f.write("*END STEP\n")

print(f"nodes={len(nodes)} elems={len(elems)} tipnodes={len(tip)}")
