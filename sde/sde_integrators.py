"""
sde/sde_integrators.py
======================
Numerical integrators for SDE trajectory data generation.

This file is an **offline data-generation tool only** — it is NOT imported at
pipeline runtime.  Run ``sde_data_gen.py`` from the command line to produce
the .npy files consumed by ``data/sde_loader.py``.

Implemented integrators
-----------------------
sw_sle_em
    Single-well potential, Stochastic Langevin Equation (underdamped),
    Euler-Maruyama scheme.

    Physical model:
        V(x) = k * x² / 2   (harmonic single-well potential)
        m·ẍ = -k·x  -  γ·ẋ  +  √(2·γ·kT) · ξ(t)

    Discretised (Euler-Maruyama):
        x(t+dt) = x(t) + v(t)·dt
        v(t+dt) = v(t) + [(-k·x(t) - γ·v(t)) / m]·dt
                        + √(2·γ·kT·dt) / m · N(0,1)

sw_gle_oe_em
    Single-well potential, Generalised Langevin Equation with a single
    exponential (Ornstein-Uhlenbeck) memory kernel, Euler-Maruyama scheme.

    Physical model (embedded as a Markovian 3-variable system):
        V(x) = k * x² / 2
        dx/dt = v
        m·dv/dt = -k·x - η
        dη/dt = -η/τ + γ·v/τ + √(2·γ·kT/τ) · ξ(t)

    Where η(t) is the coloured-friction / coloured-noise auxiliary variable
    that encodes memory with time scale τ.

    Discretised (Euler-Maruyama on all three variables):
        x(t+dt) = x(t) + v(t)·dt
        v(t+dt) = v(t) + [(-k·x(t) - η(t)) / m]·dt
        η(t+dt) = η(t) + [-η(t)/τ + γ·v(t)/τ]·dt
                        + √(2·γ·kT·dt/τ) · N(0,1)

Parameters expected in config CSV
----------------------------------
sw_sle_em.csv columns:
    k, gamma, kT, m, dt, n_steps, seed

sw_gle_oe_em.csv columns:
    k, gamma, kT, tau, m, dt, n_steps, seed

Where:
    k       — spring constant (default 1.0)
    gamma   — friction coefficient (default 1.0)
    kT      — thermal energy (k_B · T)  (default 1.0)
    tau     — memory time scale for GLE  (default 0.5)
    m       — particle mass  (default 1.0)
    dt      — integration time step (default 0.01)
    n_steps — total number of time steps per trajectory
    seed    — base RNG seed (individual trajectories offset by their index)

Output format
-------------
Both functions return (x, v) where:
    x : (n_traj, n_steps // skip)  float32  — positions after subsampling
    v : (n_traj, n_steps // skip)  float32  — velocities after subsampling
"""

from __future__ import annotations

import numpy as np


def sw_sle_em(
    n_traj: int,
    n_steps: int,
    dt: float,
    k: float,
    gamma: float,
    kT: float,
    m: float = 1.0,
    skip: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-well Stochastic Langevin Equation, Euler-Maruyama scheme.

    Parameters
    ----------
    n_traj  : Number of independent trajectories to generate.
    n_steps : Total integration steps (before subsampling).
    dt      : Integration timestep.
    k       : Spring constant of the harmonic potential V(x) = k·x²/2.
    gamma   : Friction coefficient.
    kT      : Thermal energy (k_B × Temperature).
    m       : Particle mass (default 1.0).
    skip    : Subsampling factor — keep every *skip*-th frame (default 1).
    seed    : NumPy RNG seed (default: None → non-reproducible).

    Returns
    -------
    x : (n_traj, n_steps // skip)  float32  — position trajectories.
    v : (n_traj, n_steps // skip)  float32  — velocity trajectories.
    """
    rng = np.random.default_rng(seed)
    n_out = n_steps // skip

    x_out = np.empty((n_traj, n_out), dtype=np.float32)
    v_out = np.empty((n_traj, n_out), dtype=np.float32)

    # Noise amplitude per step: √(2·γ·kT·dt) / m
    noise_amp = float(np.sqrt(2.0 * gamma * kT * dt)) / m

    # Thermal initial conditions
    x_init = rng.normal(0.0, float(np.sqrt(kT / k)),  n_traj).astype(np.float32)
    v_init = rng.normal(0.0, float(np.sqrt(kT / m)),  n_traj).astype(np.float32)

    x = x_init.copy()
    v = v_init.copy()
    out_idx = 0

    for step in range(n_steps):
        if step % skip == 0:
            x_out[:, out_idx] = x
            v_out[:, out_idx] = v
            out_idx += 1

        xi  = rng.standard_normal(n_traj).astype(np.float32)
        acc = (-k * x - gamma * v) / m   # deterministic acceleration
        x   = x + v * dt
        v   = v + acc * dt + noise_amp * xi

    return x_out, v_out


def sw_gle_oe_em(
    n_traj: int,
    n_steps: int,
    dt: float,
    k: float,
    gamma: float,
    kT: float,
    tau: float,
    m: float = 1.0,
    skip: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-well Generalised Langevin Equation with Ornstein-Uhlenbeck memory,
    Euler-Maruyama scheme (GLE-OE-EM).

    The memory kernel K(t) = (γ/τ)·exp(-t/τ) is embedded via the auxiliary
    variable η, converting the non-Markovian GLE into the Markovian system:

        dx/dt = v
        m·dv/dt = -k·x - η
        dη/dt = -η/τ + γ·v/τ + √(2·γ·kT/τ) · ξ(t)

    Parameters
    ----------
    n_traj  : Number of independent trajectories.
    n_steps : Total integration steps (before subsampling).
    dt      : Integration timestep.
    k       : Spring constant of V(x) = k·x²/2.
    gamma   : Total friction coefficient.
    kT      : Thermal energy.
    tau     : Memory time scale.
    m       : Particle mass (default 1.0).
    skip    : Subsampling factor (default 1).
    seed    : NumPy RNG seed.

    Returns
    -------
    x : (n_traj, n_steps // skip)  float32  — position trajectories.
    v : (n_traj, n_steps // skip)  float32  — velocity trajectories.
    """
    rng   = np.random.default_rng(seed)
    n_out = n_steps // skip

    x_out = np.empty((n_traj, n_out), dtype=np.float32)
    v_out = np.empty((n_traj, n_out), dtype=np.float32)

    # Noise amplitude for η update: √(2·γ·kT·dt / τ)
    noise_amp = float(np.sqrt(2.0 * gamma * kT * dt / tau))

    # Thermal initial conditions
    x   = rng.normal(0.0, float(np.sqrt(kT / k)),            n_traj).astype(np.float32)
    v   = rng.normal(0.0, float(np.sqrt(kT / m)),            n_traj).astype(np.float32)
    eta = rng.normal(0.0, float(np.sqrt(gamma * kT / tau)),  n_traj).astype(np.float32)

    out_idx = 0

    for step in range(n_steps):
        if step % skip == 0:
            x_out[:, out_idx] = x
            v_out[:, out_idx] = v
            out_idx += 1

        xi = rng.standard_normal(n_traj).astype(np.float32)

        # Euler-Maruyama update for all three variables.
        x_new   = x   + v * dt
        v_new   = v   + ((-k * x - eta) / m) * dt
        eta_new = eta + (-eta / tau + gamma * v / tau) * dt + noise_amp * xi

        x, v, eta = x_new, v_new, eta_new

    return x_out, v_out
