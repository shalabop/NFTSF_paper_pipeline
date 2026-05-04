"""
sde_integrators.py
==================
Euler-Maruyama integrators for single-well (SW) SDE systems.

Two variants are provided:

  sw_sle_em       — Standard (memoryless) Langevin equation.
                    Underdamped: tracks both position x and velocity v.

  sw_gle_oe_em    — Generalized Langevin Equation with an exponentially
                    correlated (coloured) noise source (memory kernel).
                    Uses the auxiliary Ornstein-Uhlenbeck variable z to
                    represent the memory bath.

Output convention (both functions)
-----------------------------------
  positions  : np.ndarray, shape (num_traj, recorded_steps)
  velocities : np.ndarray, shape (num_traj, recorded_steps)

Offline use only — these are NOT called at pipeline runtime.
Data generated here is consumed by the pipeline via sde_loader.py.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Single-well, no memory — Standard Langevin Equation, Euler-Maruyama
# ---------------------------------------------------------------------------

def sw_sle_em(
    total_time: float,
    time_step: float,
    x0_mean: float,
    v0_mean: float,
    zeta: float,
    m: float,
    k: float,
    x_mu: float,
    T: float,
    boltzmann: float,
    num_traj: int = 1,
    x0_std: float = 0.0,
    v0_std: float = 0.0,
    skip: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Underdamped Langevin dynamics in a harmonic (single-well) potential.

    Equations of motion:
        m dv = [-k(x - x_mu) - zeta*v] dt + sqrt(2*zeta*k_B*T) dW
        dx   = v dt

    Integrated via Euler-Maruyama; velocity updated before position
    (symplectic ordering for better energy stability).

    Parameters
    ----------
    total_time  : total simulation time
    time_step   : internal integration step dt
    x0_mean     : mean initial position
    v0_mean     : mean initial velocity
    zeta        : friction coefficient
    m           : particle mass
    k           : harmonic spring constant
    x_mu        : equilibrium position (well centre)
    T           : temperature
    boltzmann   : Boltzmann constant k_B
    num_traj    : number of independent trajectories
    x0_std      : std of initial position distribution (0 = deterministic)
    v0_std      : std of initial velocity distribution (0 = deterministic)
    skip        : record one state every this many integration steps
    seed        : random seed for reproducibility

    Returns
    -------
    positions  : (num_traj, recorded_steps)
    velocities : (num_traj, recorded_steps)
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(x0_mean, x0_std, size=num_traj) if x0_std > 0 else np.full(num_traj, x0_mean)
    v = rng.normal(v0_mean, v0_std, size=num_traj) if v0_std > 0 else np.full(num_traj, v0_mean)
    x = x.astype(np.float64)
    v = v.astype(np.float64)

    n_steps = int(total_time / time_step)
    recorded = (n_steps - 1) // skip + 1

    positions  = np.empty((num_traj, recorded), dtype=np.float32)
    velocities = np.empty((num_traj, recorded), dtype=np.float32)

    sigma = np.sqrt(2.0 * zeta * boltzmann * T)  # noise amplitude
    dt_sqrt = np.sqrt(time_step)

    positions[:, 0]  = x
    velocities[:, 0] = v
    rec_idx = 1

    for i in range(1, n_steps):
        dw  = rng.standard_normal(num_traj) * dt_sqrt
        F   = -k * (x - x_mu)
        acc = (F - zeta * v) / m
        v   = v + acc * time_step + (sigma / m) * dw
        x   = x + v * time_step
        if i % skip == 0 and rec_idx < recorded:
            positions[:, rec_idx]  = x
            velocities[:, rec_idx] = v
            rec_idx += 1

    return positions, velocities


# ---------------------------------------------------------------------------
# Single-well, with memory — Generalized Langevin Equation,
# Ornstein-Euler scheme, Euler-Maruyama
# ---------------------------------------------------------------------------

def sw_gle_oe_em(
    total_time: float,
    time_step: float,
    x0_mean: float,
    v0_mean: float,
    zeta: float,
    m: float,
    k: float,
    x_mu: float,
    T: float,
    boltzmann: float,
    tau_mem: float,
    sigma_mem: float,
    num_traj: int = 1,
    x0_std: float = 0.0,
    v0_std: float = 0.0,
    skip: int = 1,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Underdamped Generalised Langevin Equation (GLE) with exponentially
    correlated (coloured) noise, integrated via Euler-Maruyama.

    The GLE introduces an auxiliary Ornstein-Uhlenbeck process z(t) that
    acts as the memory bath (coloured noise source):

        m dv = [-k(x - x_mu) - zeta*v + z] dt
        dx   = v dt
        dz   = -z/tau_mem dt + sigma_mem/tau_mem * sqrt(2*tau_mem*k_B*T*zeta) dW

    Parameters
    ----------
    total_time  : total simulation time
    time_step   : internal integration step dt
    x0_mean     : mean initial position
    v0_mean     : mean initial velocity
    zeta        : friction coefficient (Markovian part)
    m           : particle mass
    k           : harmonic spring constant
    x_mu        : equilibrium position (well centre)
    T           : temperature
    boltzmann   : Boltzmann constant k_B
    tau_mem     : memory correlation time of the OU noise process
    sigma_mem   : coupling strength of the memory bath
    num_traj    : number of independent trajectories
    x0_std      : std of initial position distribution (0 = deterministic)
    v0_std      : std of initial velocity distribution (0 = deterministic)
    skip        : record one state every this many integration steps
    seed        : random seed for reproducibility

    Returns
    -------
    positions  : (num_traj, recorded_steps)
    velocities : (num_traj, recorded_steps)
    """
    rng = np.random.default_rng(seed)

    x = rng.normal(x0_mean, x0_std, size=num_traj) if x0_std > 0 else np.full(num_traj, x0_mean)
    v = rng.normal(v0_mean, v0_std, size=num_traj) if v0_std > 0 else np.full(num_traj, v0_mean)
    z = np.zeros(num_traj, dtype=np.float64)  # auxiliary OU variable (memory bath)
    x = x.astype(np.float64)
    v = v.astype(np.float64)

    n_steps = int(total_time / time_step)
    recorded = (n_steps - 1) // skip + 1

    positions  = np.empty((num_traj, recorded), dtype=np.float32)
    velocities = np.empty((num_traj, recorded), dtype=np.float32)

    # OU noise amplitude: ensures correct fluctuation-dissipation balance
    ou_noise_amp = sigma_mem / tau_mem * np.sqrt(2.0 * tau_mem * boltzmann * T * zeta)
    dt_sqrt = np.sqrt(time_step)

    positions[:, 0]  = x
    velocities[:, 0] = v
    rec_idx = 1

    for i in range(1, n_steps):
        dw  = rng.standard_normal(num_traj) * dt_sqrt
        # OU memory variable update
        z   = z - (z / tau_mem) * time_step + ou_noise_amp * dw
        # GLE equations of motion
        F   = -k * (x - x_mu)
        acc = (F - zeta * v + z) / m
        v   = v + acc * time_step
        x   = x + v * time_step
        if i % skip == 0 and rec_idx < recorded:
            positions[:, rec_idx]  = x
            velocities[:, rec_idx] = v
            rec_idx += 1

    return positions, velocities
