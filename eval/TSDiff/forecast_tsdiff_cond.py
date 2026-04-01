import argparse
import logging
import yaml
import numpy as np
import torch
import datetime
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import MetaData, TrainDatasets, FileDataset
from gluonts.evaluation import make_evaluation_predictions

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    add_config_to_argparser,
    filter_metrics,
    MaskInput,
)
from uncond_ts_diff.model import TSDiffCond
import uncond_ts_diff.configs as diffusion_configs

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def load_model(config: dict) -> TSDiffCond:
    """
    Initializes TSDiffCond architecture and loads checkpoint weights.
    """
    model = TSDiffCond(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq              = config["freq"],
        use_features      = config["use_features"],
        use_lags          = config["use_lags"],
        normalization     = config["normalization"],
        context_length    = config["context_length"],
        prediction_length = config["prediction_length"],
        init_skip         = config["init_skip"],
        noise_observed    = config.get("noise_observed", True),
    )

    logger.info(f"Loading from {config['ckpt']} ...")
    checkpoint = torch.load(config["ckpt"], map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model = model.to(config["device"])
    model.eval()
    logger.info(f"Loaded checkpoint from: {config['ckpt']}")
    return model


def forecast(
    config           : dict,
    model            : TSDiffCond,
    test_dataset,
    transformation,
    time             : np.ndarray,
    train_test_split : int,
) -> dict:
    """
    Runs TSDiffCond direct inference and returns standardized forecast bundle.
    """
    num_samples       = config["num_samples"]
    prediction_length = config["prediction_length"]

    transformed_testdata = transformation.apply(test_dataset, is_train=False)

    test_splitter = create_splitter(
        past_length   = config["context_length"] + max(model.lags_seq),
        future_length = prediction_length,
        mode          = "test",
    )

    # MaskInput adds orig_past_target which TSDiffCond needs
    masking_transform = MaskInput(
        FieldName.TARGET,
        FieldName.OBSERVED_VALUES,
        config["context_length"],
        "none",   # no missing values for standard forecasting
        0,
    )
    test_transform = test_splitter + masking_transform

    # TSDiffCond uses get_predictor directly — no guidance sampler
    predictor = model.get_predictor(
        test_transform,
        batch_size = 1280,
        device     = config["device"],
    )

    forecast_it, ts_it = make_evaluation_predictions(
        dataset     = transformed_testdata,
        predictor   = predictor,
        num_samples = num_samples,
    )
    

    results          = list(tqdm(forecast_it, total=len(transformed_testdata)))
    print(f"samples shape: {results[0].samples.shape}")
    print(f"num samples: {len(results[0].samples)}")
    
    #samples = np.load("results/cond_tsfdiff/double_well.npz")["samples"]
    # check variance across samples for first trajectory
    #print(samples[0, :, :].std(axis=1))  # std across 50 samples at each timestep
    ts_list          = list(ts_it)
    full_trajectories = np.array([ts.values.flatten() for ts in ts_list])

    

    forecast_samples = np.array([f.samples for f in results])
    forecast_samples = np.transpose(forecast_samples, (0, 2, 1))  # (N, T_pred, S)

    time_test  = time[train_test_split:train_test_split + prediction_length]
    time_train = time[:train_test_split]

    ci90_lower = np.percentile(forecast_samples, 5,  axis=2)
    ci90_upper = np.percentile(forecast_samples, 95, axis=2)
    ci50_lower = np.percentile(forecast_samples, 25, axis=2)
    ci50_upper = np.percentile(forecast_samples, 75, axis=2)

    # Plot 8 example forecasts
    '''fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15, 12))
    axes = axes.flatten()
    for i in range(8):
        median_forecast = np.median(forecast_samples[i], axis=1)
        axes[i].plot(time, full_trajectories[i], color='black')
        axes[i].plot(time_test, median_forecast, color='blue')
        axes[i].fill_between(time_test, ci90_lower[i], ci90_upper[i], color='blue', alpha=0.3)
        axes[i].fill_between(time_test, ci50_lower[i], ci50_upper[i], color='orange', alpha=0.3)
        axes[i].set_xlabel("Time")
        axes[i].set_ylabel("Position")
        axes[i].axhline(-1, linestyle="--", color='red')
        axes[i].axhline(1,  linestyle="--", color='red')'''

    #fig.suptitle(f'TSDiff-Cond {config["dataset"]}', fontsize=16)
    path = Path("plots/cond_tsfdiff")
    path.mkdir(parents=True, exist_ok=True)
    filename = datetime.datetime.now().strftime("cond_tsfdiff_forecasts_%Y-%m-%d_%H:%M:%S")
    plt.savefig(path / filename)
    plt.close()

    return {
        "samples"           : forecast_samples,
        "ground_truth"      : full_trajectories[:, train_test_split:train_test_split + prediction_length],
        "full_trajectories" : full_trajectories,
        "ci90_lower"        : ci90_lower,
        "ci90_upper"        : ci90_upper,
        "ci50_lower"        : ci50_lower,
        "ci50_upper"        : ci50_upper,
        "time_test"         : time_test,
        "time_train"        : time_train,
        "time"              : time,
        "train_test_split"  : train_test_split,
        "prediction_length" : prediction_length,
        "item_ids"          : np.arange(len(results)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run TSDiff-Cond forecasting and save results."
    )
    parser.add_argument("--config", "-c", required=True,
                        help="Path to forecast config yaml")
    parser.add_argument("--checkpoint", required=False,
                        help="Path to model checkpoint")
    parser.add_argument("--dataset_path", required=True,
                        help="Path to GluonTS dataset directory")
    parser.add_argument("--out", "-o", default="results/cond_tsfdiff",
                        help="Output directory or .npz path")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    if config.get("ckpt") is None:
        config["ckpt"] = args.checkpoint
    config["device"] = args.device

    logger.info(f"dataset          : {config['dataset']}")
    logger.info(f"context_length   : {config['context_length']}")
    logger.info(f"prediction_length: {config['prediction_length']}")
    logger.info(f"ckpt             : {config['ckpt']}")

    # Load dataset
    dataset_path = Path(args.dataset_path)
    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)

    metadata = MetaData(
        freq              = meta_json["freq"],
        prediction_length = meta_json["prediction_length"],
    )
    test_ds  = FileDataset(dataset_path / "test", freq=metadata.freq)
    train_ds = FileDataset(dataset_path / "train", freq=metadata.freq)
    dataset  = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    full_trajectories = np.array([entry["target"] for entry in dataset.test])

    time_npz         = np.load(dataset_path / "time.npz")
    time             = time_npz["time"]
    train_test_split = int(time_npz["train_test_split"])

    # Load model
    model = load_model(config)

    # Setup transformation
    transformation = create_transforms(
        num_feat_dynamic_real = 0,
        num_feat_static_cat   = 0,
        num_feat_static_real  = 0,
        time_features         = model.time_features,
        prediction_length     = config["prediction_length"],
    )

    # Run forecast
    logger.info("Running forecast...")
    results = forecast(
        config           = config,
        model            = model,
        test_dataset     = dataset.test,
        transformation   = transformation,
        time             = time,
        train_test_split = train_test_split,
    )

    # Save results
    out_path = Path(args.out)
    if out_path.suffix != ".npz":
        out_path.mkdir(parents=True, exist_ok=True)
        out_path = out_path / f"{config['dataset']}.npz"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_path,
        samples           = results["samples"],
        ground_truth      = results["ground_truth"],
        full_trajectories = results["full_trajectories"],
        ci90_lower        = results["ci90_lower"],
        ci90_upper        = results["ci90_upper"],
        ci50_lower        = results["ci50_lower"],
        ci50_upper        = results["ci50_upper"],
        time_test         = results["time_test"],
        time_train        = results["time_train"],
        time              = results["time"],
        train_test_split  = results["train_test_split"],
        prediction_length = results["prediction_length"],
        item_ids          = results["item_ids"],
    )

    logger.info(f"Saved to: {out_path}")
    logger.info(f"  samples shape     : {results['samples'].shape}")
    logger.info(f"  ground_truth shape: {results['ground_truth'].shape}")


if __name__ == "__main__":
    main()