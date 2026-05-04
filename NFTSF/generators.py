import torch
import numpy as np

def single_well_generator(total_time, time_step, x0_mean, zeta, k, x_mu, T,
    boltzmann, num_of_simulations=1, x0_std=0, device='cuda', sample_every=1
):
    """
    Simulates overdamped Langevin dynamics of a particle in a harmonic (single-well) potential.

    The potential energy is quadratic:
        U(x) = (k / 2) * (x - x_mu)^2

    yielding the deterministic restoring force:
        F(x) = -dU/dx = -k * (x - x_mu)

    In the overdamped (high-friction) regime, inertia is negligible, so the
    equation of motion reduces to the first-order stochastic differential
    equation (SDE):
        zeta * dx = F(x) dt + sqrt(2 * k_B * T * zeta) * dW(t)

    or equivalently:
        dx = -(k / zeta) * (x - x_mu) * dt + sqrt(2 * k_B * T / zeta) * dW(t)

    where:
        zeta    : friction (damping) coefficient
        k_B     : Boltzmann constant
        T       : temperature
        dW(t)   : Wiener process increment, dW ~ N(0, dt)

    This SDE is integrated numerically via the Euler-Maruyama scheme:
        x(t + dt) = x(t) + [-(k / zeta) * (x(t) - x_mu)] * dt
                    + sqrt(2 * k_B * T / zeta) * dW

    where dW is drawn from N(0, dt) at each time step.

    Args:
        total_time: total simulation time for each trajectory
        time_step: discretization step dt (internal integration timestep)
        x0_mean: mean of the initial position distribution
        zeta: friction (damping) coefficient
        k: harmonic force constant (spring constant)
        x_mu: equilibrium position (center of the well)
        T: temperature
        boltzmann: Boltzmann constant k_B (1.38e-23 in SI units)
        num_of_simulations: number of independent trajectories to generate
        x0_std: std of the Gaussian from which x(0) is sampled (0 = deterministic start)
        device: torch device ('cuda' or 'cpu')
        sample_every: record one state every this many integration steps (default 1 = record all).
                      The initial condition at step 0 is always recorded.

    Returns:
        positions: tensor of shape (num_of_simulations, recorded_steps)
                   where recorded_steps = (num_time_steps - 1) // sample_every + 1
        times: 1-D tensor of recorded time values, spacing = time_step * sample_every
    """

    torch.set_default_device(device)
    if x0_std == 0:
        x0 = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x0 = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,))

    num_time_steps = int(total_time / time_step)
    recorded_steps = (num_time_steps - 1) // sample_every + 1

    # Diffusion coefficient: sigma = sqrt(2 * k_B * T / zeta)
    diffusion = torch.sqrt(torch.tensor(2 * boltzmann * T / zeta, device=device))
    dt_tensor = torch.tensor(time_step, device=device)

    # Allocate only the recorded output (not the full dense trajectory)
    positions = torch.zeros((num_of_simulations, recorded_steps), device=device)
    times = torch.zeros(recorded_steps, device=device)

    # Store initial condition
    x = x0.clone()
    positions[:, 0] = x
    times[0] = 0.0
    record_idx = 1

    # Euler-Maruyama integration of the overdamped Langevin SDE:
    #   x(t+dt) = x(t) + drift * dt + diffusion * dW
    # where drift = -(k/zeta)*(x - x_mu) is the deterministic force term
    # and diffusion * dW is the stochastic thermal noise term.
    # This loop is sequential because x(t+dt) depends on x(t) through
    # the state-dependent drift (linear recurrence with coefficient != 1).
    # dW is generated per-step to avoid pre-allocating the full dense noise array.
    for i in range(1, num_time_steps):
        dw = torch.normal(mean=0.0, std=np.sqrt(time_step),
                          size=(num_of_simulations,), device=device)
        x = x + -(k / zeta) * (x - x_mu) * dt_tensor + diffusion * dw
        if i % sample_every == 0:
            positions[:, record_idx] = x
            times[record_idx] = i * time_step
            record_idx += 1

    return (positions, times)

def double_wells_generator(total_time, time_step, x0_mean, zeta, left_well, right_well,
                            barrier_height, T, boltzmann, tilt=0, num_of_simulations=1,
                            x0_std=0, device='cuda', sample_every=1):
        """
        Simulates overdamped Langevin dynamics of a particle in a symmetric
        double-well potential with an optional linear tilt.

        The double-well potential is a quartic function centered at the midpoint
        between the two wells:
            U(x) = H * [ ((x - m) / a)^4 - 2 * ((x - m) / a)^2 ] + tilt * x

        where:
            m = (left_well + right_well) / 2   (midpoint between wells)
            a = |left_well - right_well| / 2   (half-distance between wells)
            H = barrier_height                 (height of the energy barrier)

        This produces two minima at x = m +/- a (i.e., at left_well and
        right_well) separated by a barrier of height H at x = m.

        The force is the negative gradient of U(x):
            F(x) = -dU/dx = -H * [ 4*(x-m)^3 / a^4 - 4*(x-m) / a^2 ] - tilt

        The tilt parameter introduces an asymmetry: tilt > 0 tilts the
        potential to favor the left well; tilt < 0 favors the right well.

        The overdamped Langevin SDE is:
            dx = -(1/zeta) * dU/dx * dt + sqrt(2 * k_B * T / zeta) * dW(t)

        integrated via Euler-Maruyama:
            x(t+dt) = x(t) - (dU/dx / zeta) * dt + sigma * dW

        where sigma = sqrt(2 * k_B * T / zeta) and dW ~ N(0, dt).

        The loop is inherently sequential because the cubic force term
        F(x) = -dU/dx depends nonlinearly on the current position x(t),
        precluding vectorized (e.g., cumsum-based) integration.

        Args:
            total_time: total simulation time
            time_step: discretization step dt (internal integration timestep)
            x0_mean: mean initial position
            zeta: friction coefficient
            left_well: position of the left potential minimum
            right_well: position of the right potential minimum
            barrier_height: energy barrier height H between the two wells
            T: temperature
            boltzmann: Boltzmann constant k_B
            tilt: linear tilt applied to the potential (default 0, symmetric)
            num_of_simulations: number of independent trajectories
            x0_std: std of initial position distribution (0 = deterministic)
            device: torch device
            sample_every: record one state every this many integration steps (default 1 = record all).
                          The initial condition at step 0 is always recorded.

        Returns:
            positions: tensor of shape (num_of_simulations, recorded_steps)
                       where recorded_steps = (num_time_steps - 1) // sample_every + 1
            times: 1-D tensor of recorded time values, spacing = time_step * sample_every
        """

        midpoint = (left_well + right_well) / 2.0
        a = abs(left_well - right_well) / 2.0

        torch.set_default_device(device)
        if x0_std == 0:
            x0 = torch.full((num_of_simulations,), x0_mean, device=device)
        else:
            x0 = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,))

        num_time_steps = int(total_time / time_step)
        recorded_steps = (num_time_steps - 1) // sample_every + 1

        # Diffusion coefficient: sigma = sqrt(2 * k_B * T / zeta)
        diffusion = torch.sqrt(torch.tensor(2 * boltzmann * T / zeta, device=device))
        dt_tensor = torch.tensor(time_step, device=device)

        # Allocate only the recorded output (not the full dense trajectory)
        positions = torch.zeros((num_of_simulations, recorded_steps), device=device)
        times = torch.zeros(recorded_steps, device=device)

        # Store initial condition
        x = x0.clone()
        positions[:, 0] = x
        times[0] = 0.0
        record_idx = 1

        # Euler-Maruyama integration:
        #   dU/dx = H * [4*(x-m)^3 / a^4 - 4*(x-m) / a^2] + tilt
        #   x(t+dt) = x(t) - (dU/dx / zeta) * dt + sigma * dW
        # dW is generated per-step to avoid pre-allocating the full dense noise array.
        for i in range(1, num_time_steps):
            dw = torch.normal(mean=0.0, std=np.sqrt(time_step),
                              size=(num_of_simulations,), device=device)
            dUdx = barrier_height * (4 * ((x - midpoint)**3) / (a**4) - 4 * (x - midpoint) / (a**2)) + tilt
            x = x - (dUdx / zeta) * dt_tensor + diffusion * dw
            if i % sample_every == 0:
                positions[:, record_idx] = x
                times[record_idx] = i * time_step
                record_idx += 1

        return (positions, times)

def single_well_generator_underdamped(total_time, time_step, x0_mean, v0_mean, zeta, m, k, x_mu, T,
                                      boltzmann, num_of_simulations=1, x0_std=0, v0_std=0,
                                      device='cuda', sample_every=1):
    """
    Simulates underdamped Langevin dynamics of a particle in a harmonic
    (single-well) potential, retaining full inertial effects.

    Unlike the overdamped case, the particle mass m is finite, so both
    position x and velocity v are tracked. The potential is:
        U(x) = (k / 2) * (x - x_mu)^2

    The underdamped Langevin equation is a second-order SDE:
        m * dv = [ F(x) - zeta * v ] dt + sqrt(2 * zeta * k_B * T) * dW(t)
        dx = v * dt

    where:
        F(x) = -k * (x - x_mu)    (harmonic restoring force)
        zeta * v                   (viscous damping force)
        sqrt(2 * zeta * k_B * T)  (fluctuation-dissipation thermal noise amplitude)
        dW(t) ~ N(0, dt)          (Wiener process increment)

    The fluctuation-dissipation relation ensures that the noise amplitude
    sqrt(2 * zeta * k_B * T) is consistent with the damping coefficient zeta,
    so the system thermalizes to the Boltzmann distribution at temperature T.

    Euler-Maruyama integration proceeds as:
        a(t)     = [ F(x(t)) - zeta * v(t) ] / m
        v(t+dt)  = v(t) + a(t) * dt + [sqrt(2 * zeta * k_B * T) / m] * dW
        x(t+dt)  = x(t) + v(t+dt) * dt

    Note: velocity is updated first, then position uses the updated velocity
    (a symplectic-like ordering that improves energy stability).

    The loop is sequential because the acceleration depends on the coupled
    state (x(t), v(t)) at each step.

    Args:
        total_time: total simulation time
        time_step: discretization step dt (internal integration timestep)
        x0_mean: mean initial position
        v0_mean: mean initial velocity
        zeta: friction (damping) coefficient
        m: particle mass
        k: harmonic force constant
        x_mu: equilibrium position (well center)
        T: temperature
        boltzmann: Boltzmann constant k_B
        num_of_simulations: number of independent trajectories
        x0_std: std of initial position distribution (0 = deterministic)
        v0_std: std of initial velocity distribution (0 = deterministic)
        device: torch device
        sample_every: record one state every this many integration steps (default 1 = record all).
                      The initial condition at step 0 is always recorded.

    Returns:
        positions: tensor of shape (num_of_simulations, recorded_steps)
                   where recorded_steps = (num_time_steps - 1) // sample_every + 1
        velocities: tensor of shape (num_of_simulations, recorded_steps)
        times: 1-D tensor of recorded time values, spacing = time_step * sample_every
    """
    torch.set_default_device(device)

    if x0_std == 0:
        x = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,), device=device)

    if v0_std == 0:
        v = torch.full((num_of_simulations,), v0_mean, device=device)
    else:
        v = torch.normal(mean=torch.tensor(v0_mean), std=torch.tensor(v0_std), size=(num_of_simulations,), device=device)

    num_time_steps = int(total_time / time_step)
    recorded_steps = (num_time_steps - 1) // sample_every + 1

    # Allocate only the recorded output (not the full dense trajectory)
    positions = torch.zeros((num_of_simulations, recorded_steps), device=device)
    velocities = torch.zeros((num_of_simulations, recorded_steps), device=device)
    times = torch.zeros(recorded_steps, device=device)

    # Store initial condition
    positions[:, 0] = x
    velocities[:, 0] = v
    times[0] = 0.0
    record_idx = 1

    # Noise amplitude from fluctuation-dissipation: sqrt(2 * zeta * k_B * T / dt)
    # The division by sqrt(dt) is absorbed here so that multiplying by dW ~ N(0, dt)
    # yields the correct variance: Var = 2 * zeta * k_B * T * dt
    diffusion = torch.sqrt(torch.tensor(2 * zeta * boltzmann * T / time_step, device=device))
    dt_tensor = torch.tensor(time_step, device=device)

    # Euler-Maruyama integration of the coupled (x, v) system:
    #   F       = -k * (x - x_mu)               [harmonic restoring force]
    #   acc     = (F - zeta * v) / m             [net acceleration]
    #   v(t+dt) = v(t) + acc * dt + (sigma/m)*dW [velocity update]
    #   x(t+dt) = x(t) + v(t+dt) * dt           [position update with new velocity]
    # dW is generated per-step to avoid pre-allocating the full dense noise array.
    for i in range(1, num_time_steps):
        dw = torch.normal(mean=0.0, std=np.sqrt(time_step),
                          size=(num_of_simulations,), device=device)
        F = -k * (x - x_mu)
        acc = (F - zeta * v) / m
        v = v + acc * dt_tensor + diffusion / m * dw
        x = x + v * dt_tensor

        if i % sample_every == 0:
            positions[:, record_idx] = x
            velocities[:, record_idx] = v
            times[record_idx] = i * time_step
            record_idx += 1

    return positions, velocities, times


def double_wells_generator_underdamped(total_time, time_step, x0_mean, v0_mean, zeta, m,
                                       left_well, right_well, barrier_height, T, boltzmann,
                                       tilt=0, num_of_simulations=1, x0_std=0, v0_std=0,
                                       device='cuda', sample_every=1):
    """
    Simulates underdamped Langevin dynamics of a particle in a double-well
    quartic potential with an optional linear tilt, retaining full inertial effects.

    The double-well potential is:
        U(x) = H * [ ((x - mid) / a)^4 - 2 * ((x - mid) / a)^2 ] + tilt * x

    where:
        mid = (left_well + right_well) / 2   (midpoint)
        a   = |left_well - right_well| / 2   (half-distance between minima)
        H   = barrier_height                 (barrier height at the midpoint)

    The force from this potential is:
        F(x) = -dU/dx = -H * [4*(x-mid)^3/a^4 - 4*(x-mid)/a^2] - tilt

    The full underdamped Langevin equations of motion are:
        m * dv = [ F(x) - zeta * v ] dt + sqrt(2 * zeta * k_B * T) * dW(t)
        dx = v * dt

    This couples the nonlinear quartic force with viscous damping (zeta * v)
    and thermal fluctuations satisfying the fluctuation-dissipation theorem.

    Euler-Maruyama integration:
        dUdx    = H * [4*(x-mid)^3/a^4 - 4*(x-mid)/a^2] + tilt
        acc     = (-dUdx - zeta * v) / m
        v(t+dt) = v(t) + acc * dt + [sqrt(2*zeta*k_B*T) / m] * dW
        x(t+dt) = x(t) + v(t+dt) * dt

    The loop is sequential because the cubic force depends nonlinearly on x(t)
    and the velocity coupling makes it a two-variable system.

    Args:
        total_time: total simulation time
        time_step: discretization step dt (internal integration timestep)
        x0_mean: mean initial position
        v0_mean: mean initial velocity
        zeta: friction (damping) coefficient
        m: particle mass
        left_well: position of the left potential minimum
        right_well: position of the right potential minimum
        barrier_height: energy barrier height H
        T: temperature
        boltzmann: Boltzmann constant k_B
        tilt: linear tilt (>0 favors left well, <0 favors right well)
        num_of_simulations: number of independent trajectories
        x0_std: std of initial position distribution (0 = deterministic)
        v0_std: std of initial velocity distribution (0 = deterministic)
        device: torch device
        sample_every: record one state every this many integration steps (default 1 = record all).
                      The initial condition at step 0 is always recorded.

    Returns:
        positions: tensor of shape (num_of_simulations, recorded_steps)
                   where recorded_steps = (num_time_steps - 1) // sample_every + 1
        velocities: tensor of shape (num_of_simulations, recorded_steps)
        times: 1-D tensor of recorded time values, spacing = time_step * sample_every
    """
    midpoint = (left_well + right_well) / 2.0
    half_dist = abs(left_well - right_well) / 2.0

    torch.set_default_device(device)

    if x0_std == 0:
        x = torch.full((num_of_simulations,), x0_mean, device=device)
    else:
        x = torch.normal(mean=torch.tensor(x0_mean), std=torch.tensor(x0_std), size=(num_of_simulations,), device=device)

    if v0_std == 0:
        v = torch.full((num_of_simulations,), v0_mean, device=device)
    else:
        v = torch.normal(mean=torch.tensor(v0_mean), std=torch.tensor(v0_std), size=(num_of_simulations,), device=device)

    num_time_steps = int(total_time / time_step)
    recorded_steps = (num_time_steps - 1) // sample_every + 1

    # Allocate only the recorded output (not the full dense trajectory)
    positions = torch.zeros((num_of_simulations, recorded_steps), device=device)
    velocities = torch.zeros((num_of_simulations, recorded_steps), device=device)
    times = torch.zeros(recorded_steps, device=device)

    # Store initial condition
    positions[:, 0] = x
    velocities[:, 0] = v
    times[0] = 0.0
    record_idx = 1

    # Noise amplitude from fluctuation-dissipation: sqrt(2 * zeta * k_B * T / dt)
    diffusion = torch.sqrt(torch.tensor(2 * zeta * boltzmann * T / time_step, device=device))
    dt_tensor = torch.tensor(time_step, device=device)

    # Euler-Maruyama integration of the coupled (x, v) system:
    #   dUdx    = H * [4*(x-mid)^3/a^4 - 4*(x-mid)/a^2] + tilt   [quartic force gradient]
    #   acc     = (-dUdx - zeta*v) / m                             [net acceleration]
    #   v(t+dt) = v(t) + acc*dt + (sigma/m)*dW                    [velocity update]
    #   x(t+dt) = x(t) + v(t+dt)*dt                               [position update]
    # dW is generated per-step to avoid pre-allocating the full dense noise array.
    for i in range(1, num_time_steps):
        dw = torch.normal(mean=0.0, std=np.sqrt(time_step),
                          size=(num_of_simulations,), device=device)
        dUdx = barrier_height * (4 * ((x - midpoint)**3) / (half_dist**4) - 4 * (x - midpoint) / (half_dist**2)) + tilt
        acc = (-dUdx - zeta * v) / m
        v = v + acc * dt_tensor + diffusion / m * dw
        x = x + v * dt_tensor

        if i % sample_every == 0:
            positions[:, record_idx] = x
            velocities[:, record_idx] = v
            times[record_idx] = i * time_step
            record_idx += 1

    return positions, velocities, times


def linear_gaussian_generator(total_time, time_step, noise_std=0.1,
                               slope_range=(0.5, 3.0), intercept_range=(-1.0, 1.0),
                               num_of_simulations=1, device='cuda', sample_every=1):
    """
    Generates trajectories of noisy linear curves where each trajectory
    has a randomly sampled slope and intercept:

        y_i(t) = slope_i * t + intercept_i + epsilon(t)

    where:
        slope_i     ~ Uniform(slope_range[0], slope_range[1])
        intercept_i ~ Uniform(intercept_range[0], intercept_range[1])
        epsilon(t)  ~ N(0, noise_std^2)   i.i.d. at each time step

    This is NOT a stochastic differential equation or Langevin dynamics.
    It serves as a simple baseline landscape for testing the normalizing
    flow model on data with known, predictable linear structure plus
    additive Gaussian noise.

    Because this generator evaluates an analytic function of time rather
    than integrating an SDE, sample_every is handled by directly computing
    values on the recorded timeline (every sample_every-th time index),
    rather than simulating a dense hidden trajectory.

    Args:
        total_time: total simulation time (determines time range)
        time_step: discretization step dt (internal integration timestep)
        noise_std: standard deviation of Gaussian noise at each step
        slope_range: (min, max) tuple for uniform slope sampling
        intercept_range: (min, max) tuple for uniform intercept sampling
        num_of_simulations: number of independent trajectories
        device: torch device ('cuda' or 'cpu')
        sample_every: record one point every this many time_step intervals (default 1 = record all).
                      Recorded time spacing = time_step * sample_every.

    Returns:
        positions: tensor of shape (num_of_simulations, recorded_steps)
                   where recorded_steps = (num_time_steps - 1) // sample_every + 1
        times: 1-D tensor of recorded time values, spacing = time_step * sample_every
    """
    torch.set_default_device(device)
    num_time_steps = int(total_time / time_step)
    recorded_steps = (num_time_steps - 1) // sample_every + 1

    # Compute times directly on the recorded (subsampled) timeline.
    # Index i in recorded output corresponds to internal step i * sample_every,
    # which has physical time i * sample_every * time_step.
    times = torch.arange(recorded_steps, device=device).float() * (time_step * sample_every)

    # Random slope and intercept per trajectory
    slopes = torch.empty(num_of_simulations, 1, device=device).uniform_(
        slope_range[0], slope_range[1])
    intercepts = torch.empty(num_of_simulations, 1, device=device).uniform_(
        intercept_range[0], intercept_range[1])

    # Deterministic linear trend: slope_i * t + intercept_i
    positions = slopes * times.unsqueeze(0) + intercepts

    # Additive i.i.d. Gaussian noise
    noise = torch.normal(mean=0.0, std=noise_std,
                         size=(num_of_simulations, recorded_steps), device=device)
    positions = positions + noise

    return (positions, times)
