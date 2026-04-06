import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions

import get_dw_dataset
from pathlib import Path

from bin.guidance_experiment import load_model
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.utils import create_transforms, create_splitter, MaskInput
from uncond_ts_diff.sampler import DDPMGuidance

CKPT_PATH = "lightning_logs/version_0/best_checkpoint.ckpt"

with open("configs/guidance/guidance_dw.yaml") as f:
    config = yaml.safe_load(f)

#NUM_SAMPLES = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device
config["ckpt"] = CKPT_PATH

model = load_model(config)
model.to(device)
model.eval()

NUM_SAMPLES = config['num_samples']

dataset =get_dw_dataset.get_dw_dataset(Path('./bin/synthetic_datasets1/test/data.json'),freq='H',prediction_length=500) 
test_dataset = dataset.test

transformation = create_transforms(
    num_feat_dynamic_real=0,
    num_feat_static_cat=0,
    num_feat_static_real=0,
    time_features=model.time_features,
    prediction_length=config["prediction_length"],
)

past_length = config["context_length"] + max(model.lags_seq)

test_splitter = create_splitter(
    past_length=past_length,
    future_length=config["prediction_length"],
    mode="test",
)

masking = MaskInput(
    FieldName.TARGET,
    FieldName.OBSERVED_VALUES,
    config["context_length"],
    missing_scenario="none",
    missing_values=0,
)

test_transform = test_splitter + masking

sampler = DDPMGuidance(
    model=model,
    prediction_length=config["prediction_length"],
    num_samples=NUM_SAMPLES,
    missing_scenario="none",
    missing_values=0,
    **config["sampler_params"],
)

predictor = sampler.get_predictor(
    test_transform,
    batch_size=1280 // NUM_SAMPLES,
    device=device,
)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=transformation.apply(test_dataset, is_train=False),
    predictor=predictor,
    num_samples=NUM_SAMPLES,
)

forecast = next(forecast_it)
ts = next(ts_it)

# -----------------------------
# Data preparation
# -----------------------------
trajectory = np.asarray(ts.values).squeeze()          # true past + future
future_samples = forecast.samples                     # (N, T_future)
future_mean = np.asarray(forecast.mean).squeeze()

T_total = len(trajectory)
T_future = future_samples.shape[1]
T_past = T_total - T_future

lower_90 = np.percentile(future_samples, 5, axis=0)
upper_90 = np.percentile(future_samples, 95, axis=0)

lower_50 = np.percentile(future_samples, 25, axis=0)
upper_50 = np.percentile(future_samples, 75, axis=0)

plt.figure(figsize=(12, 5))

plt.plot(np.arange(T_total), trajectory, color="black", linewidth=2, label="True Trajectory")

#NUM_DISPLAY = min(50, NUM_SAMPLES=1)
NUM_DISPLAY=1
#for i in range(NUM_DISPLAY):
    #plt.plot(np.arange(T_past, T_total), future_samples[i], color="royalblue", alpha=0.15, linewidth=1)

import numpy as np
from sklearn.metrics import mean_absolute_error

# ground truth for the forecast horizon
ground_truth = trajectory[T_past:]           # shape (T_future,)
# mean prediction across all samples
forecast_mean = np.mean(future_samples, axis=0)  # shape (T_future,)

# compute MAE
mae = mean_absolute_error(ground_truth, forecast_mean)

plt.fill_between(np.arange(T_past, T_total), lower_90, upper_90, color="blue", alpha=0.25, label="90% Confidence Interval")
plt.fill_between(np.arange(T_past, T_total), lower_50, upper_50, color="red", alpha=0.25, label="50% Confidence Interval")

plt.plot(np.arange(T_past, T_total), future_mean, color="red", linestyle="--", linewidth=2, label="Predicted Mean")
plt.axvline(T_past - 1, linestyle="--", color="gray")

plt.title(f"Guided Diffusion Forecast (position) — {mae}")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
#plt.ylim(-200, 250)
#plt.xlim(400,500)
plt.tight_layout()
#plt.savefig("dw3.png", dpi=150)
plt.savefig("dw1.png", dpi=150)
plt.show()

train_dataset = dataset.train
count = 0
for _ in train_dataset:
    count += 1

print("Number of training time series:", count)

'''
all_ground_truths = []
all_lower_90 = []
all_upper_90 = []

import itertools

print('----------------------------------------------------')
for forecast, ts in itertools.islice(zip(forecast_it, ts_it), 1000):

    trajectory = np.asarray(ts.values).squeeze()
    future_samples = forecast.samples

    T_total = len(trajectory)
    T_future = future_samples.shape[1]
    T_past = T_total - T_future

    ground_truth = trajectory[T_past:]

    lower_90 = np.percentile(future_samples, 5, axis=0)
    upper_90 = np.percentile(future_samples, 95, axis=0)

    all_ground_truths.append(ground_truth)
    all_lower_90.append(lower_90)
    all_upper_90.append(upper_90)

all_ground_truths = np.stack(all_ground_truths)
all_lower_90 = np.stack(all_lower_90)
all_upper_90 = np.stack(all_upper_90)

def fractionMetric(trajectories, ci_lower, ci_upper):

    trajectories = torch.tensor(trajectories)
    ci_lower = torch.tensor(ci_lower)
    ci_upper = torch.tensor(ci_upper)

    inside = (trajectories >= ci_lower) & (trajectories <= ci_upper)

    # mean over series dimension
    fraction = inside.float().mean(dim=0)

    return fraction

ci90_time=fractionMetric(all_ground_truths, lower_90, upper_90)
ci50_time=fractionMetric(all_ground_truths, lower_50, upper_50)

plt.figure(figsize=(12, 5))
plt.plot(np.arange(T_past, T_total),ci50_time.detach().cpu().numpy(),color="blue", linewidth=2, label="CI50")
plt.plot(np.arange(T_past, T_total),ci90_time.detach().cpu().numpy(),color="red",  linewidth=2, label="CI90")

plt.title(f"Fraction withing CI")
plt.xlabel("Time")
plt.ylabel("Fraction")
plt.legend()
#plt.ylim(-200, 250)
#plt.xlim(400,500)
plt.tight_layout()
#plt.savefig("dw3.png", dpi=150)
plt.savefig("fractions.png", dpi=150)
plt.show()
print(ground_truth.shape)'''