import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap, design
S=fem.Solver()
T9="../2026-11-14-metamaterial-inverse-design/results/"
cands={k:v for k,v in snap.snap_starts().items() if k in('h6_t1.9_a2','h5_t1.6_a2','hinge_h5_t1.5_b-0.4','cos_h5_t1.6','fresh_0','fresh_3')}
cands['t9_final']=np.array(json.load(open(T9+"opt_bistable_attempt6_C.json"))["across_ridge"]["p"])
cands['AO_success']=np.array(json.load(open('results/designs/bistable.json'))['p'])
for n,p in cands.items():
    X=S.coords(jnp.asarray(p)); F,zs,_=S.path(p,snap.DS)
    idx=snap.indices(F); i2=idx[2] if idx else len(F)-1
    lam=np.array([S.min_eig(zs[k],snap.DS[k],X,1)[0] for k in range(0,i2+1,2)])
    k=int(np.argmin(lam)); print(n,'min lam %.2e at d=%.2f'%(lam[k],snap.DS[2*k]),'lam at d=0.25: %.2e'%lam[0],'median %.2e'%np.median(lam), 'a2',p[1])
