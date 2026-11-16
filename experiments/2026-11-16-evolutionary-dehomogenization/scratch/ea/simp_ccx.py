import sys, json, numpy as np
sys.path.insert(0, '.')
import verify_ccx
d = np.load('results/simp/simp_designs.npz')
out = []
for i, vf in enumerate(d['vf']):
    if round(float(vf), 2) not in (0.30, 0.35, 0.40, 0.45): continue
    r = verify_ccx.verify(d['phi'][i], kind='phi', r_c=0.0, mesh_size=0.5, buckling=True, threads=4, tag=f'simp{i}')
    out.append(dict(vf=float(vf), ccx_C=float(r['compliance']), ccx_blf=float(r['blf']), ok=bool(r['ok'])))
    print(out[-1], flush=True)
json.dump(out, open('results/simp/simp_ccx.json', 'w'), indent=1)
