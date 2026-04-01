import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.loader import InferenceDataLoader
from gluonts.torch.batchify import batchify

from uncond_ts_diff.model.diffusion.tsdiff import TSDiff
from uncond_ts_diff.utils import load_config

RESULT_YAML = "results/guidance_logs/results-0.yaml"

with open(RESULT_YAML) as f:
    cfg = yaml.safe_load(f)["config"]

ckpt_path = cfg["ckpt"]
device = torch.device(cfg["device"])

context_length = cfg["context_length"]
prediction_length = cfg["prediction_length"]
num_samples = cfg["num_samples"]
dataset_name = cfg["dataset"]

train_cfg = load_config(cfg["config"])

model_cfg = train_cfg["model"]
diffusion_cfg = train_cfg["diffusion"]

model = TSDiff.load_from_checkpoint(
    ckpt_path,
    backbone_parameters=model_cfg["backbone_parameters"],
    timesteps=diffusion_cfg["timesteps"],
    diffusion_scheduler=diffusion_cfg["scheduler"],
    context_length=context_length,
    prediction_length=prediction_length,
    freq=train_cfg["dataset"]["freq"],
    normalization=train_cfg["dataset"]["normalization"],
    use_features=train_cfg["dataset"]["use_features"],
    use_lags=train_cfg["dataset"]["use_lags"],
)

model.to(device)
model.eval()

dataset = get_dataset(dataset_name, regenerate=False)
test_ds = dataset.test


loader = InferenceDataLoader(
    test_ds,
    batch_size=1,
    stack_fn=batchify,
)

batch = next(iter(loader))
batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

with torch.no_grad():
    forecasts = model.predict_step(
        batch,
        batch_idx=0,
        num_samples=num_samples,
    )

# forecasts shape: (num_samples, batch, prediction_length)
samples = forecasts[:, 0, :].cpu().numpy()

ast_target = batch["past_target"][0, -context_length:].cpu().numpy()

t_past = np.arange(context_length)
t_future = np.arange(context_length, context_length + prediction_length)


plt.figure(figsize=(12, 5))

# Past
plt.plot(t_past, past_target, color="black", label="Past")

# Sampled futures
for s in samples:
    plt.plot(t_future, s, color="blue", alpha=0.1)

# Mean + intervals
mean = samples.mean(axis=0)
q10, q90 = np.quantile(samples, [0.1, 0.9], axis=0)

plt.plot(t_future, mean, color="red", label="Mean forecast")
plt.fill_between(t_future, q10, q90, color="red", alpha=0.2, label="10–90% interval")

plt.axvline(context_length - 1, linestyle="--", color="gray")
plt.legend()
plt.title("Conditional diffusion forecast")
plt.xlabel("Time")
plt.ylabel("Value")

plt.tight_layout()
plt.show()
