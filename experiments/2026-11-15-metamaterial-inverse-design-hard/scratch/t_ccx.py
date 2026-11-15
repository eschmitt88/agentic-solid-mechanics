import sys,json,time
sys.path.insert(0,'.')
import numpy as np
import ccx, design
T9="../2026-11-14-metamaterial-inverse-design/results/"
v=json.load(open(T9+"verify_T2.json")); p=np.array(v['p'])
t=time.time(); d,F,info=ccx.run("test_T2_t3",p,4,lc_div=3.0,threads=2); print(info,time.time()-t)
sig=ccx.sample(d,F,design.travel_grid(1),info['completed'])*design.KPA
print('max diff vs trial9 ccx_t3 kPa',np.max(np.abs(sig-np.array(v['ccx_t3']))), design.grade('T2',sig))
# stack test: T2 design under cosine arch
import fem
p2=np.concatenate([p,fem.cosine_params(3.0,1.2)])
t=time.time(); d,F,info=ccx.run("test_stack",p2,4,ncell=2,dmax=20.0,ninc=80,lc_div=3.0,threads=2); print({k:v for k,v in info.items() if k!='d_cellA'},time.time()-t)
S2=fem.Solver(ncell=2); D2=np.arange(1,41)*0.5
F2,z2,i2=S2.path(p2,D2)
Fc=ccx.sample(d,F,D2,True)
print('stack: max |F_ccx-F_jax|',np.nanmax(np.abs(Fc-F2)),'range',F2.max()-F2.min(),'completed',info['completed'])
dA=np.array([S2.cell_travels(z,D)[0] for z,D in zip(z2,D2)])
print('dA jax',np.round(dA[::8],3),'ccx',np.round(np.interp(D2[::8],d,info['d_cellA']),3))
