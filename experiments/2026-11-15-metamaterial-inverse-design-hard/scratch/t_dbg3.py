import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"])
S=fem.Solver()
for dh in (-1e-4,1e-4):
    q=p.copy(); q[5]+=dh
    X=S.coords(jnp.asarray(q)); ev=[]
    F,zs,infos=S.path(q,snap.DS,events=ev)
    print('dh',dh,'events',[(e['d0'],e['d1'],round(e['jump'],4)) for e in ev])
    for k in range(22,34):
        lam=S.min_eig(zs[k],snap.DS[k],X,2)
        print('  ',snap.DS[k],F[k],'it',infos[k]['iters'],'sh',infos[k]['shifted'],'ref',infos[k].get('refined',0),'lam',lam)
    # fine continuation through the zone
    dfine=np.round(np.arange(3.0,3.8001,0.005),6)
    F2,z2,i2=S.path(q,dfine,u0=zs[23],d_start=float(snap.DS[23]))
    lam=[S.min_eig(z,d,X,1)[0] for z,d in zip(z2[::8],dfine[::8])]
    print('  fine lam min', np.round(np.array(lam)*1e6,3))
    print('  fine F at 3.5', F2[list(dfine).index(3.5)], 'coarse', F[27], 'sh', sum(i['shifted'] for i in i2))
