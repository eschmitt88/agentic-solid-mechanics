import sys,json,time
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
P=snap.FoldProblem(log=open('/dev/null','w'))
c=P.state_features(snap.snap_starts()['h6_t1.9_a2'])
cands=dict(snap.snap_starts())
for f,k in [("opt_bistable_attempt5_B","h6_t1.9_a2"),("opt_bistable_attempt3_stalled","h5.5_t2.0_a2"),("opt_bistable_attempt5_A","hinge_h5.5_t1.4_b-0.7"),("opt_bistable_attempt4_hinge_stalled","hinge_h5_t1.1_b-0.8"),("opt_bistable_attempt6_C","across_ridge")]:
    cands["end_"+k]=np.array(json.load(open(T9+f+".json"))[k]["p"])
for n,p in cands.items():
    t=time.time(); c=P.state_features(p)
    sc=snap.indices(c['F']) if np.all(np.isfinite(c['F'])) else None
    g=None if sc is None else snap.features(c['F'],sc)[0]
    print(n,'feat',None if c['feat'] is None else np.round(c['feat'],5),'grid',None if g is None else np.round(g,5),'sw',c['switched'],'events',c.get('n_events'),'snaps',[(round(f['d_s'],4),round(f['loss'],6),f['exact']) for f in c.get('snaps',[])],round(time.time()-t,1),'s',flush=True)
