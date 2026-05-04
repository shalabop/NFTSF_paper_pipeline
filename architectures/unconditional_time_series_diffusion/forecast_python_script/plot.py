import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import torch

import yaml
from bin.guidance_experiment import load_model

CKPT_PATH = "lightning_logs/version_42130616/best_checkpoint.ckpt" # Path to model checkpoint in lightning_logs/version_*/
with open("configs/guidance/guidance_uber_tlc.yaml") as f: 
    config = yaml.safe_load(f) #converts yaml to dictionary

RESULTS_DIR = Path("results/guidance_logs") # Directory to save results
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

model = load_model(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

NUM_SAMPLES = 100  
all_samples = []

from uncond_ts_diff.dataset import get_gts_dataset
dataset=  get_gts_dataset(config['dataset'])

train_data= dataset.train
test_data= dataset.test

print(train_data)

'''for entry in dataset:
    past = entry["past_target"]
    past_len = past.shape[-1]

    # Generate forecasts
    samples = model.sample_n(num_samples=NUM_SAMPLES, return_lags=False)
    # Concatenate past + forecast for plotting
    full_series = np.concatenate([past, samples], axis=-1)
    all_samples.append(full_series)

all_samples = np.stack(all_samples)  # shape: (num_series, num_samples, total_len)
np.save(RESULTS_DIR / "samples.npy", all_samples)
print("Saved forecast samples to:", RESULTS_DIR / "samples.npy")'''
