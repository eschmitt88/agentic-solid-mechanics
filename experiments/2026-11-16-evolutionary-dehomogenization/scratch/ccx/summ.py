import sys, json
for l in open(sys.argv[1] if len(sys.argv) > 1 else "ccx_checks.log"):
    if not l.startswith('{'): continue
    r = json.loads(l)
    if r.get('blf') is None: print(r['design'], r['r_c'], r['mesh_size'], 'FAILED', r.get('reason')); continue
    print(r['design'], r.get('kind'), r['r_c'], r['mesh_size'], 'C=%.3f BLF=%.4f modes=%s mode@%s pr=%.4f vf=%.4f nel=%d eps=%.3f/%.3f t=%.0f' % (r['compliance'], r['blf'], [round(x,4) for x in r['blf_modes']], [round(x,2) for x in r.get('mode_xy') or []], r.get('mode_pr') or -1, r['vf_ccx'], r['n_elements'], r['eps_shape'], r['eps_seg'], r['wall']))
