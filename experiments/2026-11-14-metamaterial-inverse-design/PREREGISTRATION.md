# Pre-registration — written before any solver or optimiser was run

Written 2026-11-14 20:37 UTC, before `fem.py` existed. Copied verbatim into
`README.md` ("Pass criteria"). Nothing below may be edited after the first
optimisation run; misses are reported as misses.

## Unit cell (fixed)

One elastomer curved beam spanning a stiff frame, loaded at mid-span through a
stiff shuttle (the classic negative-stiffness / phase-transforming cell).

- span between wall faces 60 mm; shuttle width 4 mm (each half-beam 28 mm long);
  cell width W = 70 mm; cell height H = 25 mm; plane strain, unit depth (1 mm).
- frame and shuttle are rigid; frames stay parallel and do not shear
  (uniaxial compression of a lattice between platens).
- material: compressible neo-Hookean in CalculiX's form
  U = C10 (J^-2/3 I1 - 3) + (1/D1)(J-1)^2, C10 = 0.5 MPa, D1 = 0.428571 /MPa
  (mu = 1 MPa, K = 4.6667 MPa, small-strain E = 2.8 MPa, nu = 0.4,
  plane-strain modulus E' = E/(1-nu^2) = 3.3333 MPa).
- nominal stress sigma = F / W (F per unit depth), nominal strain eps = d / H,
  d = downward shuttle travel, 0 <= eps <= 0.5.
- design parameters (9): centre-line y_c = sum_{k=1..4} a_k (1 - cos(2 pi k x/l))/2
  (zero slope at walls and shuttle), thickness
  t = t0 exp(sum_{m=1..4} b_m cos(2 pi m x / l)); symmetric about mid-span.
  Bounds: 0.4 mm <= t <= 3 mm everywhere, |y_c| <= 9 mm.

## Rung 0 — forward-solver validation (cosine beam, Qiu 2003/2004)

Reference: J. Qiu, PhD thesis, MIT 2003, ch. 2, eqs (2.9)-(2.48)
(https://pergatory.mit.edu/research/jqiu/thesisscreen.pdf), the source of
Qiu, Lang & Slocum, JMEMS 13(2) 2004, doi:10.1109/JMEMS.2004.825308.
Second-mode-constrained cosine beam, Q = h/t >= 6:
f_top ~ 740 E I h / l^3, f_bot ~ 370 E I h / l^3, d_top ~ 0.16 h,
d_bot ~ 1.92 h, d_end ~ 1.99 h (eqs 2.47-2.48). In plane strain E -> E'.

Pass: JAX solver f_top and f_bot within 5 % of eq (2.47); d_top, d_bot, d_end
within 0.05 h of eq (2.48); and the JAX curve within 2 % NRMSE of a CalculiX
solve of the same beam. Test beam: l = 100 mm, t = 1 mm, h = 6 mm (Q = 6).

## Rung 1 — target curves (sigma in kPa, eps = d/H)

Defined from generic shapes, not from any design in the parameter space.

- T1 monotonic stiffening (J-curve): sigma = a (exp(5 eps) - 1),
  a = 2 / (exp(2.5) - 1) = 0.17884 kPa  (sigma(0.5) = 2.0 kPa).
- T2 buckling-type plateau: PCHIP through
  (0,0) (0.05,0.30) (0.10,0.40) (0.35,0.40) (0.42,0.55) (0.50,1.00).
- T3 snap-through (non-monotonic): PCHIP through
  (0,0) (0.07,0.60) (0.32,0.15) (0.50,0.90).

Error metric: NRMSE = sqrt(mean_k (sigma_k - sigma*_k)^2) / (max sigma* - min sigma*)
on the 50 strains eps_k = 0.01, 0.02, ..., 0.50.

Pass (judged on the CalculiX curve only):
- T1: NRMSE <= 0.05 and CalculiX curve monotonic.
- T2: NRMSE <= 0.05 and |sigma - 0.40| <= 0.06 kPa (15 %) for all 0.12 <= eps <= 0.33.
- T3: NRMSE <= 0.08 and the CalculiX curve has an interior local maximum
  followed by a local minimum (a negative-stiffness branch).

Verifier: CalculiX 2.23, NLGEOM, same neo-Hooke constants, full beam (no symmetry
imposed), body-fitted unstructured 6-node triangles (CPE6) with element size
<= t_min/3, shuttle displaced vertically with no rotation and no lateral motion.
Additional, reported but not part of pass/fail: laterally free shuttle.

Sanity check (labelled as such, not a result): recover a known in-space design.

## Rung 2 — programmed bistability (stretch)

Targets (per mm depth): snap-through peak force F_pk = 0.040 N/mm,
second stable position d_2 = 10.0 mm, energy barrier for returning from the
second state E_back = integral of -F over [d_u, d_2] = 0.045 N mm/mm
(d_u = unstable equilibrium).
Pass: each of the three within 5 % in CalculiX, and both equilibria shown
stable (tangent positive-definite with the shuttle force-free in the design
solver; CalculiX holds the second state with the shuttle released).
