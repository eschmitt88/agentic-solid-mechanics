import sys,json
sys.path.insert(0,'.')
import numpy as np, jax.numpy as jnp
import fem, snap
T9="../2026-11-14-metamaterial-inverse-design/results/"
p=np.array(json.load(open(T9+"opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"])
P=snap.FoldProblem(log=open('/dev/null','w'))
S=P.S
for dh in (-1e-4,-3e-5,0,3e-5,1e-4):
    q=p.copy(); q[6]+=dh
    X=S.coords(jnp.asarray(q)); ev=[]
    F,zs,infos=S.path(q,snap.DS,events=ev)
    c=P.state_features(q)
    print('dh',dh,'E',c['feat'][2],'Eend',c['E_endpoints'],'du',c['d_u'],'d2',c['feat'][1],'nsnap',c['nsnap'],[ (round(f['d_s'],5),round(f['loss'],7),f['exact'],round(f['jump'],3)) for f in c['snaps']])
    for e in ev:
        d0,z0,d1,z1=P.folds.bracket(X,e)
        print('   event',e['d0'],e['d1'],'jump',e['jump'],'after bracket',d0,d1,float(jnp.max(jnp.abs(z1-z0))), 'E drop', 2*float(S.energy(z0,d0,X)-S.energy(z1,d1,X)))
