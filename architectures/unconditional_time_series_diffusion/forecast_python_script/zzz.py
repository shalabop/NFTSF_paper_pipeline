import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions

from bin.guidance_experiment import load_model
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.utils import create_transforms, create_splitter, MaskInput
from uncond_ts_diff.sampler import DDPMGuidance

CKPT_PATH = "lightning_logs/version_42800599/best_checkpoint.ckpt"

with open("configs/guidance/guidance_electricity.yaml") as f:
    config = yaml.safe_load(f)

NUM_SAMPLES = 300
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["device"] = device
config["ckpt"] = CKPT_PATH

model = load_model(config)
model.to(device)
model.eval()

dataset = get_gts_dataset(config["dataset"])
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

# Convert to 1D arrays
trajectory = np.asarray(ts.values).squeeze()
future_pred_mean = np.asarray(forecast.mean).squeeze()
future_samples = forecast.samples 

T_total = len(trajectory)
T_future = len(future_pred_mean)
T_past = T_total - T_future

plt.figure(figsize=(12,5))

for i in range(NUM_SAMPLES):
    plt.plot(
        np.arange(T_past, T_total),
        future_samples[i],
        color="blue",
        alpha=0.1
            )
plt.plot(np.arange(T_total), trajectory, color="black", linewidth=2, label="True Trajectory")

# Predicted future (overlay)
#plt.plot(np.arange(T_past, T_total), future_pred_mean, color="red", linestyle="--", linewidth=2, label="Predicted Future")


plt.axvline(T_past-1, linestyle="--", color="gray")  # separator
plt.title("Guided Diffusion Forecast (Electricity)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.ylim(-200,250)
plt.tight_layout()
plt.xlim(3900,4000)
plt.savefig("electricity_forecast.png", dpi=150)
plt.close()

