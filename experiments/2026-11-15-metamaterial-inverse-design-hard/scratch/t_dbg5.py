import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt6_C.json"))["across_ridge"]["p"])
P=snap.FoldProblem(log=sys.stdout)
for h in (-1e-4,-2e-5,0,2e-5,1e-4):
    q=p.copy(); q[0]+=h
    c=P.state_features(q)
    X=P.S.coords(jnp.asarray(q)); ev=[]
    P.S.path(q,snap.DS,events=ev)
    print(h,'E',c['feat'][2],'Eend',c['E_endpoints'],'du',c['d_u'],'nsnap',c['nsnap'],[(round(s['d_s'],5),round(s['loss'],7),round(s['jump'],3),s['exact']) for s in c['snaps']])
    for e in ev:
        d0,z0,d1,z1=P.folds.bracket(X,e)
        print('    ev',round(e['d0'],4),round(e['d1'],4),round(e['jump'],4),'->',d0,d1,round(float(jnp.max(jnp.abs(z1-z0))),5),'Edrop',2*float(P.S.energy(z0,d0,X)-P.S.energy(z1,d1,X)))
