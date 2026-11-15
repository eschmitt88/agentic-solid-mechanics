import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"])
S=fem.Solver()
R={}
for dh in (0,3e-5):
    q=p.copy(); q[6]+=dh
    X=S.coords(jnp.asarray(q)); ev=[]
    F,zs,infos=S.path(q,snap.DS,events=ev)
    R[dh]=(F,zs,infos,X)
F0,z0,i0,X0=R[0]; F1,z1,i1,X1=R[3e-5]
for k in range(len(F0)):
    dz=float(jnp.max(jnp.abs(z0[k]-z1[k])))
    if dz>1e-3 or i0[k].get('refined') or i1[k].get('refined') or i0[k]['switched'] or i1[k]['switched']:
        lam0=S.min_eig(z0[k],snap.DS[k],X0,1)[0]; lam1=S.min_eig(z1[k],snap.DS[k],X1,1)[0]
        print(snap.DS[k],'F',F0[k],F1[k],'dz',round(dz,4),'ref',i0[k].get('refined',0),i1[k].get('refined',0),'sw',i0[k]['switched'],i1[k]['switched'],'jump',i0[k].get('jump'),i1[k].get('jump'),'lam',lam0,lam1,'sh',i0[k]['shifted'],i1[k]['shifted'])
