import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from gluonts.dataset.field_names import FieldName
from gluonts.evaluation import make_evaluation_predictions

from bin.guidance_experiment import load_model
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.utils import create_transforms, create_splitter, MaskInput
from uncond_ts_diff.sampler import DDPMGuidance

CKPT_PATH = "lightning_logs/version_42800230/best_checkpoint.ckpt"

with open("configs/guidance/guidance_uber_tlc.yaml") as f:
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

test_splitter = create_splitter(
    past_length=config["context_length"] + max(model.lags_seq),
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

for i, (forecast, ts) in enumerate(zip(forecast_it, ts_it)):
    past = ts.values
    future_samples = forecast.samples
    median_forecast = np.median(future_samples, axis=0)
    lower = np.percentile(future_samples, 5, axis=0)
    upper = np.percentile(future_samples, 95, axis=0)
    
    T_past = len(past)
    T_future = future_samples.shape[1]
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.arange(T_past), past, color="blue", label="Ground Truth")
    plt.plot(np.arange(T_past, T_past + T_future), median_forecast, color="red", label="Median Forecast")
    plt.fill_between(np.arange(T_past, T_past + T_future), lower, upper, color="red", alpha=0.3, label="90% Interval")
    plt.axvline(T_past - 1, linestyle="--", color="gray", label="Forecast Horizon")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(f"Forecast for Series {i}")
    plt.legend()
    plt.tight_layout()
    
    Path("forecast_plots").mkdir(exist_ok=True)
    plt.savefig(f"forecast_plots/forecast_{i}.png", dpi=150)
    plt.close()

