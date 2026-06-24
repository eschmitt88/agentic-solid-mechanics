"""Finite-strain viscohyperelastic material-point model for uniaxial VHB 4910:
an Ogden equilibrium network in parallel with one Bergstrom-Boyce (BB) viscous
branch. End-to-end differentiable in JAX, so model parameters can be calibrated
to experimental stress-stretch histories by gradient descent.

Incompressible uniaxial kinematics: axial stretch lambda(t) prescribed, lateral
stretches lambda^-1/2. The viscous branch splits the axial stretch
multiplicatively lambda = lambda_e * lambda_v; its elastic part lambda_e drives
a (neo-Hookean) overstress, and lambda_v flows per the BB rule. Stress is
reported as NOMINAL (engineering) stress P = force/undeformed-area, the
convention of Hossain et al. (2012); P = sigma_Cauchy / lambda for uniaxial.

Parameters (physical), packed/unpacked from an unconstrained vector for the
optimizer (positives via softplus):
    Ogden N=2 equilibrium : mu1, alpha1, mu2, alpha2
    BB viscous branch     : muV (branch stiffness), eta (flow stress scale),
                            m (flow stress exponent)
    fixed                 : C = -1.0, xi = 0.01  (classic BB constants)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import lax

jax.config.update("jax_enable_x64", True)

C_BB = -1.0     # BB chain-stretch exponent (classic value)
XI = 0.01       # BB regularization (avoids singularity at chain stretch 1)
N_FP = 6        # semi-implicit fixed-point iterations per step
NPARAM = 7


# --------------------------------------------------------------------------- #
# parameter (de)serialization: unconstrained vector <-> physical params
# --------------------------------------------------------------------------- #
def _softplus(x):
    return jnp.logaddexp(x, 0.0)


def unpack(theta):
    """theta: (7,) unconstrained -> dict of positive/real physical params."""
    return {
        "mu1": _softplus(theta[0]),
        "alpha1": theta[1],
        "mu2": _softplus(theta[2]),
        "alpha2": theta[3],
        "muV": _softplus(theta[4]),
        "eta": _softplus(theta[5]) + 1e-6,
        "m": _softplus(theta[6]) + 1.0,   # m >= 1
    }


# --------------------------------------------------------------------------- #
# stresses (nominal / engineering, uniaxial incompressible)
# --------------------------------------------------------------------------- #
def ogden_nominal(lam, p):
    """Nominal uniaxial stress of the 2-term Ogden equilibrium network."""
    def term(mu, a):
        return mu * (lam ** (a - 1.0) - lam ** (-a / 2.0 - 1.0))
    return term(p["mu1"], p["alpha1"]) + term(p["mu2"], p["alpha2"])


def branch_cauchy(lam_e, muV):
    """Cauchy overstress of the neo-Hookean viscous-branch spring (uniaxial)."""
    return muV * (lam_e ** 2 - 1.0 / lam_e)


def _flow_rate(sigma_B, lam_v, p):
    """BB viscous logarithmic-strain rate magnitude (>=0)."""
    lam_chain = jnp.sqrt((lam_v ** 2 + 2.0 / lam_v) / 3.0)
    soft = (jnp.maximum(lam_chain - 1.0 + XI, XI)) ** C_BB
    drive = (jnp.abs(sigma_B) / p["eta"]) ** p["m"]
    return soft * drive


# --------------------------------------------------------------------------- #
# time integration of the viscous internal variable (differentiable scan)
# --------------------------------------------------------------------------- #
def stress_history(lam_t, dt_t, theta):
    """Predicted nominal-stress history for a prescribed stretch history.
    lam_t, dt_t: (T,) arrays. Returns (T,) nominal stress."""
    p = unpack(theta)

    def step(lv_prev, inp):
        lam, dt = inp
        # semi-implicit: solve lv = lv_prev + dt * rate(lam, lv) * sign(sigma_B)
        def fp(lv):
            lam_v = jnp.exp(lv)
            lam_e = lam / lam_v
            sB = branch_cauchy(lam_e, p["muV"])
            rate = _flow_rate(sB, lam_v, p)
            return lv_prev + dt * rate * jnp.sign(sB)
        lv = lv_prev
        for _ in range(N_FP):
            lv = 0.5 * lv + 0.5 * fp(lv)      # damped fixed point
        lam_v = jnp.exp(lv)
        lam_e = lam / lam_v
        sB = branch_cauchy(lam_e, p["muV"])
        P = ogden_nominal(lam, p) + sB / lam   # nominal = Cauchy / total stretch
        return lv, P

    _, P_t = lax.scan(step, 0.0, (lam_t, dt_t))
    return P_t


@jax.jit
def predict(lam_t, dt_t, theta):
    return stress_history(lam_t, dt_t, theta)
