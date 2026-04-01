import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
import logging
import tqdm

import data_generation.linear.linear_process as linear_process
import data_generation.langevin.generators as langevin

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def generate_gaussian_linear(sim_cfg: dict,device=torch.device):
    
    total_time  = sim_cfg["total_time"]
    time_step   = sim_cfg["time_step"]
    
    num_of_simulations      = sim_cfg["num_of_simulations"]
    steps = int(total_time / time_step)
    
    positions,time =linear_process.linear_gaussian_generator(time_step=time_step,
                                                             total_time=total_time,
                                                             num_of_simulations=num_of_simulations)

    return positions.detach().cpu().numpy(), time.detach().cpu().numpy()

def generate_linear(sim_cfg: dict, device: torch.device):
    """
    Generate linear AR1 trajectories.
    Returns positions (N, T) as numpy, time (T,) as numpy.
    """
    total_time  = sim_cfg["total_time"]
    time_step   = sim_cfg["time_step"]
    slope_mean       = sim_cfg["slope_mean"]
    slope_std       = sim_cfg["slope_std"]
    intercept   = sim_cfg["intercept"]
    noise       = sim_cfg["noise"]
    phi         = sim_cfg["phi"]
    num_of_simulations      = sim_cfg["num_of_simulations"]

    steps = int(total_time / time_step)
    time_steps = [time_step * i for i in range(steps)]

    positions =linear_process.linear(
            trajectories=num_of_simulations,
            steps=time_steps,
            slope_mean=slope_mean,
            slope_std=slope_std,
            intercept=intercept,
            noise=noise,
            phi=phi,
        )                                      
    time = np.array(time_steps)                
    return positions, time


def generate_single_well(sim_cfg: dict, device: torch.device):
    """
    Generate single-well harmonic oscillator trajectories via Langevin dynamics.
    Returns positions (N, T) as numpy, time (T,) as numpy.
    """
    boltzmann = sim_cfg["boltzmann"]
    zeta      = sim_cfg["zeta"]
    T         = sim_cfg["temperature"]
    k         = sim_cfg["k"]
    x_mu      = sim_cfg["x_mu"]
    x0_mean   = sim_cfg["x0_mean"]
    x0_std    = sim_cfg.get("x0_std", 0)
    n_sims    = sim_cfg["num_of_simulations"]
    total_time = sim_cfg["total_time"]
    time_step  = sim_cfg["time_step"]

    positions, times = langevin.single_well_generator(
        total_time=total_time,
        time_step=time_step,
        x0_mean=x0_mean,
        x0_std=x0_std,
        zeta=zeta,
        k=k,
        x_mu=x_mu,
        T=T,
        boltzmann=boltzmann,
        num_of_simulations=n_sims,
        device=device,
    )

    positions = positions.detach().cpu().numpy()   
    times     = times.detach().cpu().numpy()  
    return positions, times


def generate_double_well(sim_cfg: dict, device: torch.device):
    """
    Generate double-well Langevin trajectories.
    Returns positions (N, T) as numpy, time (T,) as numpy.
    """
    boltzmann      = sim_cfg["boltzmann"]
    zeta           = sim_cfg["zeta"]
    T              = sim_cfg.get("temperature", 1/boltzmann)
    left_well      = sim_cfg["left_well"]
    right_well     = sim_cfg["right_well"]
    tilt           = sim_cfg.get("tilt", 0)
    x0_mean        = sim_cfg["x0_mean"]
    x0_std         = sim_cfg.get("x0_std", 0)
    n_sims         = sim_cfg["num_of_simulations"]
    total_time     = sim_cfg["total_time"]
    time_step      = sim_cfg["time_step"]
    bimodal      = sim_cfg.get("bimodal",False)

    barrier_height = sim_cfg.get("barrier_height") or 2 * boltzmann * T
    
    positions, times = langevin.double_wells_generator(
        total_time=total_time,
        time_step=time_step,
        x0_mean=x0_mean,
        x0_std=x0_std,
        zeta=zeta,
        T=T,
        boltzmann=boltzmann,
        left_well=left_well,
        right_well=right_well,
        barrier_height=barrier_height,
        tilt=tilt,
        num_of_simulations=n_sims,
        device=device,
        bimodal=bimodal
    )

    positions = positions.detach().cpu().numpy()   
    times     = times.detach().cpu().numpy()   
    return positions, times

def generate_sine(sim_cfg: dict, device: torch.device,seed=1212):
    """
    Generates a multi-frequency seasonal signal with exponential 
    frequency drift and Gaussian noise.
    """
    num_of_trajectories=sim_cfg["num_of_simulations"]
    total_time=sim_cfg["total_time"]
    time_step=sim_cfg["time_step"]
    size=int(total_time/time_step)
    time = np.arange(size)*time_step

    rng = np.random.default_rng(seed)
    xi = rng.standard_normal(size=(size, num_of_trajectories))
    
    phi1 = 2 * np.pi * rng.random(num_of_trajectories)
    t = np.arange(size).reshape(-1, 1)
    
    w1 = 2 * np.pi / 24 
    w2 = w1 * rng.exponential(size=num_of_trajectories) / 2

    positions = 4 * np.sin(phi1 + w1 * t) + np.sin(w2 * t) + xi    
    return positions.astype(np.float32).T,time


GENERATORS = {
    "linear":      generate_linear,
    "l": generate_linear,
    "single_well": generate_single_well,
    "sw": generate_single_well,
    "double_well": generate_double_well,
    "dw": generate_double_well,
    "sine" : generate_sine,
    "s" : generate_sine,
    "gl" : generate_gaussian_linear,
    "linear_gaussian" : generate_gaussian_linear,
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic time-series data and save to .npz"
    )
    parser.add_argument(
        "--data-type", "-d",
        required=True,
        choices=list(GENERATORS.keys()),
        help="Type of data to generate: linear, l | single_well, sw | double_well, dw | sine, s",
    )
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/data/double_well.yaml)",
    )
    parser.add_argument(
        "--out", "-o",
        required=False,
        help="Output path for .npz file (e.g. data/double_well.npz)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device to use for Langevin generators (default: cuda if available)",
    )

    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    seed = 12  
    torch.manual_seed(seed)
    np.random.seed(seed) 
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cpu") 
    logger.info(f"Random seed: {seed}")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    sim_cfg      = cfg["simulation"]
    pipeline_cfg = cfg["pipeline"]

    logger.info(f"Using device: {device}")

    generator_fn = GENERATORS[args.data_type]
    logger.info(f"Generating '{args.data_type}' data ...")
    logger.info(f'{generator_fn} Function Running ...')
    positions, time = generator_fn(sim_cfg, device)
    
    logger.info(f'Testing Seed Stability ...')
    logger.info(f"Generating '{args.data_type}' test data ...")
    torch.manual_seed(seed)
    np.random.seed(seed) 
    torch.cuda.manual_seed_all(seed)
    
    ##ensures data is constant across runs
    test_positions, test_time = generator_fn(sim_cfg, device)
    
    try:
        backup_data = np.load(f"training_data/backup/{args.data_type}.npz")
        backup_pos=backup_data["positions"]
        
        if np.all(np.equal(test_positions,positions)) and np.all(np.equal(backup_pos,positions)):
            logger.info("SUCCESS: test_positions == positions")
            logger.info("SUCCESS: backedup == positions")
        else:
            if not torch.eq(test_positions,positions).all():
                raise SystemError("ERROR: test_positions != positions")
            if not torch.eq(backup_pos,positions).all():
                raise SystemError("ERROR: backedup != positions")
    except:
        pass

    logger.info(f"positions shape : {positions.shape}")   
    logger.info(f"time shape      : {time.shape}")        

    if args.out is None:
        args.out=f"data/{args.data_type}"
    out_path = Path(args.out)

    if out_path.suffix == ".npy":
        out_path = out_path.with_suffix(".npz")
        logger.warning(f"Changing output extension to .npz → {out_path}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Saving file...")
    with tqdm.tqdm(total=1, desc="Saving", unit="file") as pbar:
        np.savez(
            out_path,
            positions=positions,
            time=time,
            train_test_split=pipeline_cfg["train_test_split"],
            prediction_length=pipeline_cfg["prediction_length"],
            seed=seed,
            val_start=pipeline_cfg["val_start"]
        )
        pbar.update(1)

    logger.info(f"Saved to: {out_path}")
    logger.info(f"  num trajectories : {positions.shape[0]}")
    logger.info(f"  total time steps : {time.shape[0]}")
    logger.info(f"  train steps      : {pipeline_cfg['train_test_split']}")
    logger.info(
        f"  test steps      : {time.shape[0] - pipeline_cfg['train_test_split']}"
    )
    logger.info(
        f"  prediction length: {pipeline_cfg['prediction_length']}"
    )

if __name__ == "__main__":
    main()