import yaml
import torch
from uncond_ts_diff.dataset import get_gts_dataset
from uncond_ts_diff.utils import create_transforms, create_splitter, MaskInput
from uncond_ts_diff.sampler import DDPMGuidance

from src.uncond_ts_diff.model import TSDiff  # import TSDiff
from src.uncond_ts_diff.configs import diffusion_config  # import diffusion configs

def load_model(config):
    diff_cfg_name = config.get("diffusion_config", "diffusion_small_config")
    diff_cfg = diffusion_config[diff_cfg_name]

    model = TSDiff(
        **diff_cfg,
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization="mean",
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        init_skip=config["init_skip"],
    )

    ckpt = torch.load(config["ckpt"], map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    model.load_state_dict(state_dict, strict=False)
    model.to(config["device"])
    model.eval()
    return model

with open("configs/guidance/guidance_electricity_100.yaml") as f:
    config = yaml.safe_load(f)

config["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config["ckpt"] = "lightning_logs/version_102/checkpoints/last.ckpt"

model = load_model(config)

#ckpt = torch.load(CKPT_PATH, map_location="cpu")
#state_dict = ckpt["state_dict"]  # only the model weights
#print(state_dict)

#model.load_state_dict(state_dict, strict=False)

#model.to(device)
#model.eval()

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

# ----------------------------
# 5. Create sampler/predictor
# ----------------------------
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

# ----------------------------
# 6. Make predictions
# ----------------------------
forecast_it, ts_it = make_evaluation_predictions(
    dataset=transformation.apply(test_dataset, is_train=False),
    predictor=predictor,
    num_samples=NUM_SAMPLES,
)

forecast = next(forecast_it)
ts = next(ts_it)

past = ts.values
future_samples = forecast.samples

T_past = len(past)
T_future = future_samples.shape[1]

# ----------------------------
# 7. Plot past + predicted future samples
# ----------------------------
plt.figure(figsize=(12, 5))

# Past
plt.plot(np.arange(T_past), past, color="black", linewidth=2, label="Past")

# Future samples
for i in range(NUM_SAMPLES):
    plt.plot(
        np.arange(T_past, T_past + T_future),
        future_samples[i],
        color="blue",
        alpha=0.15,
    )

plt.axvline(T_past - 1, linestyle="--", color="gray")
plt.title("Guided Diffusion Forecast (Conditional)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig("forecast_electricity.png", dpi=150)
plt.close()

print("Forecast completed and saved as forecast_electricity.png")
