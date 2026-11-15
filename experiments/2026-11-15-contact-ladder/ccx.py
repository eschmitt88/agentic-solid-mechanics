"""CalculiX helpers: deck writing from (X, quads), running, frd/dat parsing."""
import os, subprocess, time, re
import numpy as np

MM = [os.path.expanduser("~/.local/bin/micromamba"), "run", "-n", "solidmech", "ccx"]
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "results", "ccx_work")


def run(job, threads=4, timeout=3600):
    env = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"), OMP_NUM_THREADS=str(threads))
    t0 = time.time()
    try:
        r = subprocess.run(MM + [job], cwd=WORK, env=env, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr; rc = r.returncode
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""); rc = -9
    open(os.path.join(WORK, job + ".log"), "w").write(out)
    return rc, out, time.time() - t0


def nodes_block(X):
    return "*NODE, NSET=NALL\n" + "".join(f"{i+1},{x:.12g},{y:.12g}\n" for i, (x, y) in enumerate(X))


def elements_block(quads, etype, elset, start=0):
    return f"*ELEMENT, TYPE={etype}, ELSET={elset}\n" + "".join(
        f"{start+i+1},{a+1},{b+1},{c+1},{d+1}\n" for i, (a, b, c, d) in enumerate(quads))


def surface_block(name, el, face):
    return f"*SURFACE, NAME={name}, TYPE=ELEMENT\n" + "".join(f"{e+1},S{f+1}\n" for e, f in zip(el, face))


def nset(name, nodes):
    nodes = [int(n) + 1 for n in nodes]
    return f"*NSET, NSET={name}\n" + "".join(",".join(map(str, nodes[i:i + 12])) + "\n" for i in range(0, len(nodes), 12))


def parse_frd(path):
    """returns list of dict(step_time, name, comps, nodes, values) for every result block."""
    out = []; cur = None; t = None
    with open(path) as f:
        for line in f:
            k = line[:5]
            if k == "  100":
                t = float(line[12:24])
            elif k == " -4  ":
                cur = dict(time=t, name=line[5:13].strip(), nodes=[], values=[])
                out.append(cur)
            elif k == " -1  " or line[:3] == " -1":
                if cur is not None:
                    cur["nodes"].append(int(line[3:13]))
                    vals = line[13:].rstrip("\n")
                    cur["values"].append([float(vals[i:i + 12]) for i in range(0, len(vals), 12)])
            elif line[:3] == " -3":
                if cur is not None:
                    cur["nodes"] = np.array(cur["nodes"]) - 1; cur["values"] = np.array(cur["values"])
                cur = None
    return out


def parse_dat_totals(path):
    """total force blocks: returns list of (set, time, [fx, fy, fz])."""
    out = []
    lines = open(path).read().splitlines()
    for i, l in enumerate(lines):
        m = re.search(r"total force \(fx,fy,fz\) for set (\S+) and time\s+(\S+)", l)
        if m:
            j = i + 1
            while not lines[j].strip():
                j += 1
            out.append((m.group(1), float(m.group(2)), [float(v) for v in lines[j].split()]))
    return out


def summarize_log(out):
    """count increments / iterations / cutbacks from ccx stdout."""
    return dict(increments=out.count("increment size="), iterations=len(re.findall(r"^\s*iteration \d+", out, flags=re.M)),
                divergence=out.count("divergence"), errors=[l.strip() for l in out.splitlines() if "*ERROR" in l][:5],
                finished=("Job finished" in out))
