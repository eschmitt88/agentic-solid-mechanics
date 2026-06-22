import sys

L = 1.0
b = float(sys.argv[1])  # width (y)
h = float(sys.argv[2])  # height (z)
nx, ny, nz = 24, 1, 4   # elements

# C3D20 serendipity: node at grid (i,j,k) exists if #odd coords <= 1
Ni, Nj, Nk = 2*nx, 2*ny, 2*nz

def odd_count(i, j, k):
    return (i % 2) + (j % 2) + (k % 2)

nid = {}
nodes = []
n = 0
for i in range(Ni+1):
    for j in range(Nj+1):
        for k in range(Nk+1):
            if odd_count(i, j, k) <= 1:
                n += 1
                nid[(i, j, k)] = n
                x = i/Ni*L
                y = j/Nj*b
                z = k/Nk*h
                nodes.append((n, x, y, z))

# Abaqus C3D20 local offsets
offs = [
    (0,0,0),(2,0,0),(2,2,0),(0,2,0),
    (0,0,2),(2,0,2),(2,2,2),(0,2,2),
    (1,0,0),(2,1,0),(1,2,0),(0,1,0),
    (1,0,2),(2,1,2),(1,2,2),(0,1,2),
    (0,0,1),(2,0,1),(2,2,1),(0,2,1),
]

elems = []
e = 0
for ex in range(nx):
    for ey in range(ny):
        for ez in range(nz):
            base = (2*ex, 2*ey, 2*ez)
            conn = [nid[(base[0]+o[0], base[1]+o[1], base[2]+o[2])] for o in offs]
            e += 1
            elems.append((e, conn))

# node sets
fixed = [nid[(0,j,k)] for i in range(1) for j in range(Nj+1) for k in range(Nk+1) if (0,j,k) in nid]
tip = [nid[(Ni,j,k)] for j in range(Nj+1) for k in range(Nk+1) if (Ni,j,k) in nid]

with open("beam.inp", "w") as f:
    f.write("*NODE, NSET=NALL\n")
    for nn, x, y, z in nodes:
        f.write(f"{nn}, {x:.8e}, {y:.8e}, {z:.8e}\n")
    f.write("*ELEMENT, TYPE=C3D20R, ELSET=EALL\n")
    for ee, conn in elems:
        vals = [str(ee)] + [str(c) for c in conn]
        f.write(", ".join(vals[:16]) + ",\n")
        f.write(", ".join(vals[16:]) + "\n")
    def wrset(name, lst):
        f.write(f"*NSET, NSET={name}\n")
        for i in range(0, len(lst), 12):
            f.write(", ".join(str(x) for x in lst[i:i+12]) + "\n")
    wrset("FIX", fixed)
    wrset("TIP", tip)
    f.write("*MATERIAL, NAME=STEEL\n*ELASTIC\n2.1e11, 0.3\n")
    f.write("*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL\n")
    f.write("*STEP\n*STATIC\n")
    f.write("*BOUNDARY\nFIX, 1, 3, 0.0\n")
    P = 2000.0
    f.write("*CLOAD\n")
    fper = -P/len(tip)
    for t in tip:
        f.write(f"{t}, 3, {fper:.8e}\n")
    f.write("*NODE PRINT, NSET=TIP\nU\n")
    f.write("*EL PRINT, ELSET=EALL\nS\n")
    f.write("*END STEP\n")

print(f"b={b} h={h} nodes={len(nodes)} elems={len(elems)} tipnodes={len(tip)}")
euler = 2000.0*L**3/(3*2.1e11*(b*h**3/12))
print(f"Euler delta = {euler*1000:.4f} mm")
print(f"stress = {6*2000*L/(b*h**2)/1e6:.2f} MPa")
print(f"mass = {7850*L*b*h:.4f} kg")
