# agentic-solid-mechanics

**Can an LLM agent operate solid-mechanics simulators — write/run/debug FEM
decks and drive differentiable inverse-design loops — well enough to do useful
engineering analysis autonomously?**

📂 **[Browse this repo →](https://<owner>.github.io/agentic-solid-mechanics/)** —
interactive, always-live view of experiments, concepts, literature, and maps of
content. Served via GitHub Pages from `docs/index.html`; reads the live file
tree, no build step. _(Link is live once the repo is public and Pages is enabled
— `/new-project` does both by default.)_

## What this is

Solid mechanics (and, later, the multiphysics commonly coupled to it — heat
transfer, fluids, electromagnetics) is a domain where correctness is checkable
against physics, mesh convergence, and analytical benchmarks. That makes it a
strong testbed for **agentic simulation**: an LLM agent that sets up a finite
element problem, runs a solver, reads the results, diagnoses errors, and
iterates toward a goal — with the agent's competence graded against ground
truth rather than self-reported confidence.

Two agentic loops are in scope:

1. **Agent as solver operator** — the agent authors and edits solver input
   (Python API or text decks), runs it on this machine, parses results, and
   debugs failures, iterating toward an analysis or design target.
2. **Differentiable / inverse design** — the agent orchestrates gradient-based
   design optimization and parameter inversion on a differentiable solver,
   using the GPU.

Trials start with single-physics solid mechanics on this Linux server
(RTX 5080, 32 cores, 91 GB RAM) and a free / low-cost solver chosen for clean
agent ergonomics. Coupled multiphysics is a later expansion. Success = the
agent reaches correct, mesh-converged, benchmark-validated answers
autonomously, and we can characterize where and why it fails.

## How it's organized

Plain Markdown + flat YAML frontmatter, cross-linked with `[[wikilinks]]`:

- `concepts/` / `mocs/` — atomic ideas; promoted to a map of content when ≥5 cluster.
- `literature/` — processed notes on papers, repos, posts (0–5 relevance scored).
- `experiments/YYYY-MM-DD-<slug>/` — self-contained runs (hypothesis → result, config, metrics, log).
- `raw/` — immutable source captures · `docs/decisions/` — ADRs · `_meta/` — index, log, templates.

## Local use

```sh
make env    # uv sync
make lint   # knowledge-graph / experiment health check
```

Part of a personal research framework
([claude-system](https://github.com/eschmitt88/claude-system)). See `CLAUDE.md`
for the agent-facing orientation and `~/.claude/CLAUDE.md` for the framework's
durable principles.
