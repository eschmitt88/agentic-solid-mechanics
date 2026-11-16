"""hf.py on the saved real designs at h = 0.25 and at h = 0.125 (same pixel geometry, each pixel split 2x2)."""
import numpy as np
from common import hf, hf_eval_mode
for i in (0, 240, 479):
    phi = np.load(f"real_{i}.npz")["phi"]
    for h, f in ((0.25, phi), (0.125, np.kron(phi, np.ones((2, 2), bool)))):
        r = hf_eval_mode(hf.Problem(hf.GIRDER, h), f)
        print(i, h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
