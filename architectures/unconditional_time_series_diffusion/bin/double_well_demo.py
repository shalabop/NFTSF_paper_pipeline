import logging
import argparse
import yaml
import torch
import os
from pathlib import Path
from tqdm.auto import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# GluonTS Imports
from gluonts.dataset.loader import TrainDataLoader
from gluonts.dataset.split import OffsetSplitter, split
from gluonts.dataset.common import MetaData, TrainDatasets, FileDataset
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.field_names import FieldName

# TSDiff Imports
import uncond_ts_diff.configs as diffusion_configs
from uncond_ts_diff.model import TSDiff
from uncond_ts_diff.sampler import DDPMGuidance, DDIMGuidance
from uncond_ts_diff.utils import (
    create_transforms, create_splitter, add_config_to_argparser,
    filter_metrics, MaskInput
)

import matplotlib.pyplot as plt
import torch
import numpy as np
#import models.unconditional_time_series_diffusion
import metrics
import data_generation.linear.linear_process as linear_process
import data_generation.langevin.generators as langevin
import pandas as pd
#import utils 
import diffusion_model_utils.datasets
import diffusion_model_utils.config
import pmdarima as pm
import properscoring as ps

# Setup Logger
logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

guidance_map = {"ddpm": DDPMGuidance, "ddim": DDIMGuidance}

# --- Helper Functions ---

def get_custom_dataset(jsonl_path, freq, prediction_length, split_offset=None):
    if split_offset is None:
        split_offset = -prediction_length
    metadata = MetaData(freq=freq, prediction_length=prediction_length)
    test_ts = FileDataset(jsonl_path, freq)
    train_ts, _ = split(test_ts, offset=split_offset)
    return TrainDatasets(metadata=metadata, train=train_ts, test=test_ts)

def create_model(config):
    model = TSDiff(
        **getattr(diffusion_configs, config["diffusion_config"]),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization=config["normalization"],
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        lr=config["lr"],
        init_skip=config["init_skip"],
    )
    return model

def evaluate_and_save_forecasts(config, model, test_dataset, transformation, log_dir):
    logger.info("Running Evaluation/Guidance...")
    model.to(config["device"])
    model.eval()
    
    Guidance = guidance_map[config["sampler"]]
    sampler = Guidance(
        model=model,
        prediction_length=config["prediction_length"],
        num_samples=config["num_samples"],
        missing_scenario="none",
        missing_values=0,
        **config["sampler_params"],
    )

    test_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=config["prediction_length"],
        mode="test",
    )
    test_transform = test_splitter + MaskInput(FieldName.TARGET, FieldName.OBSERVED_VALUES, config["context_length"], "none", 0)
    
    predictor = sampler.get_predictor(test_transform, batch_size=32, device=config["device"])
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=transformation.apply(test_dataset, is_train=False),
        predictor=predictor,
        num_samples=config["num_samples"],
    )

    forecast_list = list(tqdm(forecast_it, total=len(test_dataset), desc="Forecasting"))
    ts_list = list(ts_it)
    
    all_forecast_samples = np.array([f.samples for f in forecast_list])
    
    all_targets = np.array([ts.values for ts in ts_list])
    
    evaluator = Evaluator()
    metrics, _ = evaluator(ts_list, forecast_list)
    metrics = filter_metrics(metrics)
    
    with open(Path(log_dir) / "eval_results.yaml", "w") as f:
        yaml.dump(metrics, f)
    
    np.save(Path(log_dir) / "forecast_samples.npy", all_forecast_samples)
    np.save(Path(log_dir) / "targets.npy", all_targets)
    
    logger.info(f"Results and NumPy arrays saved to {log_dir}")

    return metrics, all_forecast_samples, all_targets

# --- Main Execution ---

def main(log_dir):
    #-----------------------------------------# check that CUDA is mounted #-----------------------------------------#
    print("CUDA device name:", torch.cuda.is_available())
    print(torch.version.cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #-----------------------------------------# Double well  #-----------------------------------------#
    boltzman= 1.38e-11
    zeta = 2.24e-9

    T = 300
    D = (T*boltzman)/zeta
    print(f'diffusion constant: {D}')
    t0=0
    total_time_dw = 1
    time_step_dw = 4e-3

    num_of_simulations_dw=4#0
    steps_dw=int(total_time_dw/time_step_dw)

    print(f'total time steps: {steps_dw}') 

    barrier_height=2*boltzman*T
    left_well=3
    right_well=-3
    tilt=0

    double_well_data,double_well_time = langevin.double_wells_generator(total_time=total_time_dw, 
                                zeta=zeta,
                                T=T,time_step=time_step_dw,   
                                boltzmann=boltzman,
                                right_well=right_well,left_well=left_well,
                                barrier_height=barrier_height,
                                x0_mean=0,device=device  ,
                                num_of_simulations=num_of_simulations_dw,
                                tilt=tilt)

    double_well_data,double_well_time=double_well_data.detach().cpu().numpy(),double_well_time.detach().cpu().numpy()
    num_of_arima_models=num_of_simulations_dw
    linear_data_ar1_ensemble=[double_well_data[i] for i in range(num_of_arima_models)]

    #-----------------------------------------# split & plot #-----------------------------------------#
    split=100
    prediction_length=50
    time_steps_train=double_well_time[:split]
    time_steps_test=double_well_time[split:split+prediction_length]
    #prediction_length=len(time_steps_test)

    diffusion_model_utils.datasets.generate_dataset_split(dataset_path="./datasets/double_well",
          positions=double_well_data, train_test_split=split, prediction_length=prediction_length)

    print(f'total steps={steps_dw}')
    print(f'training set cardinality={len(time_steps_train)}')
    print(f'testing set cardinality={len(time_steps_test)}')
    num_of_arima_simulations = num_of_simulations_dw
    diffusion_model_utils.config.write_training_config(
        output_path="./tsf_diff_training_config/train_double_well.yaml",
                                                   dataset='double_well', 
                                                   prediction_length=prediction_length,
                                                   context_length=32,
                                                   num_samples=num_of_arima_simulations,
                                                   max_epochs=5,#0,
                                                   batch_size=128,
                                                   learning_rate=1e-2,)

    #-----------------------------------------# test ARIMA model #-----------------------------------------#
    
    linear_data_ar1_ensemble_series = [
        pd.Series(data[:split]) 
        for data in linear_data_ar1_ensemble
    ]
    models=[pm.auto_arima(training_data_series) for training_data_series in linear_data_ar1_ensemble_series]
    results=[model.arima_res_ for model in models]

    predictions = [result.get_forecast(steps=prediction_length) for result in results]

    ci90_linear_ar1 = [prediction.conf_int(alpha=0.1) for prediction in predictions] #90% ci
    ci90_linear_ar1_indices = [series[split:split+prediction_length].index for series in linear_data_ar1_ensemble_series]
    ci90_linear_ar1_lower=[ci90_lower_bound.iloc[:, 0] for ci90_lower_bound in ci90_linear_ar1] 
    ci90_linear_ar1_upper=[ci90_lower_bound.iloc[:, 1] for ci90_lower_bound in ci90_linear_ar1]

    ci50_linear_ar1 = [prediction.conf_int(alpha=0.5) for prediction in predictions] #50% ci
    ci50_linear_ar1_indices = [series[split:split+prediction_length].index for series in linear_data_ar1_ensemble_series]
    ci50_linear_ar1_lower=[ci50_lower_bound.iloc[:, 0] for ci50_lower_bound in ci50_linear_ar1] 
    ci50_linear_ar1_upper=[ci50_lower_bound.iloc[:, 1] for ci50_lower_bound in ci50_linear_ar1]


    ground_truths = double_well_data[:, split:split+prediction_length]
   
    simulations_list = []
    all_arimas_time_crps_linear_ar1=[]

    for i in range(num_of_arima_models):
        simulations_list.append(results[i].simulate(nsimulations=prediction_length, repetitions=num_of_arima_simulations,
                                           anchor='end').values)

    simulations_pt = torch.stack([torch.tensor(sim, dtype=torch.float32) for sim in simulations_list], dim=0)

    ####################################ARIMA CRPS
    for i in range(num_of_arima_models):
        score_per_step = metrics.crps_per_time(
            ground_truth=ground_truths[i], 
            predictions=simulations_pt[i]
        )

        all_arimas_time_crps_linear_ar1.append(score_per_step)

    average_crps_time = torch.stack(all_arimas_time_crps_linear_ar1).mean(dim=0)

    all_arimas_time_crps_linear_ar1_ps=[]
    for i in range(num_of_arima_models):
        score_per_step_ps = metrics.crps_per_time_ps(
            ground_truth=ground_truths[i], 
            predictions=simulations_pt[i]
        )
    
        all_arimas_time_crps_linear_ar1_ps.append(torch.tensor(score_per_step_ps))
    average_crps_time_ps = torch.stack(all_arimas_time_crps_linear_ar1_ps).mean(dim=0)

    #----------------------------------# CI #--------------------------------------#
    all_fractions_90 = []

    for i in range(num_of_arima_models):
        gt_single = ground_truths[i]

        f = metrics.fractionMetric(
            trajectories=torch.tensor(gt_single), 
            ci_lower=ci90_linear_ar1_lower[i], 
            ci_upper=ci90_linear_ar1_upper[i]
        )
        all_fractions_90.append(f)

    average_coverage_90 = torch.stack(all_fractions_90).mean(dim=0)

    all_fractions_50 = []

    for i in range(num_of_arima_models):
        gt_single = ground_truths[i]

        f = metrics.fractionMetric(
            trajectories=torch.tensor(gt_single), 
            ci_lower=ci50_linear_ar1_lower[i], 
            ci_upper=ci50_linear_ar1_upper[i]
        )
        all_fractions_50.append(f)

    average_coverage_50 = torch.stack(all_fractions_50).mean(dim=0)

    ############################### ARIMA MeAE
    arima_predictions = simulations_pt.detach().cpu().numpy()


    '''arima_mae_time=[]
    for i in range(num_of_arima_models):
        mae_time = metrics.median_absolute_error_time(
            ground_truths[i].reshape(1,-1), 
            arima_predictions[i]
        )
        arima_mae_time.append(mae_time)'''

    #---------------------------------# Training #---------------------------------#
    with open("./tsf_diff_training_config/train_double_well.yaml", "r") as f:
        config = yaml.safe_load(f)
    config['device'] = device # Ensure device is set


    # 2. Get the GluonTS Dataset (generated by generate_dataset_split)
    dataset_name="double_well"
    dataset_path = Path(f"./datasets/{dataset_name}")

    with open(dataset_path / "metadata.json", "r") as f:
        meta_json = yaml.safe_load(f)
    metadata = MetaData(freq=meta_json["freq"], prediction_length=meta_json["prediction_length"])
    train_ds = FileDataset(dataset_path / "train", freq=metadata.freq)
    test_ds = FileDataset(dataset_path / "test", freq=metadata.freq)
    dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

    # 3. Create Model & Data Loader
    model = create_model(config)
    transformation = create_transforms(
        num_feat_dynamic_real=0, num_feat_static_cat=0, num_feat_static_real=0,
        time_features=model.time_features, prediction_length=prediction_length
    )
    training_splitter = create_splitter(
        past_length=config["context_length"] + max(model.lags_seq),
        future_length=prediction_length, mode="train"
    )

    transformed_data = transformation.apply(dataset.train, is_train=True)
    data_loader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=config["batch_size"],
        stack_fn=batchify,
        transform=training_splitter,
        num_batches_per_epoch=config["num_batches_per_epoch"],
    )

    #for trainng optim 
    torch.set_float32_matmul_precision('high')

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator="auto",
        devices=1,
        default_root_dir=log_dir
    )

    #
    trainer.fit(model, train_dataloaders=data_loader)

    metrics_object, all_forecast_samples, all_targets = evaluate_and_save_forecasts(
        config, model, dataset.test, transformation, log_dir
    )

    print(f'TSF diff forecasts: {all_forecast_samples.shape}' )
    print(f'TSF diff target: {all_targets.shape}' )


    idx = 0 
    
    full_truth = double_well_data[idx]
    x_axis = np.arange(len(full_truth))
    

    tsdiff_samples = all_forecast_samples[idx]
    tsdiff_median = np.median(tsdiff_samples, axis=0)
    tsdiff_mean = np.mean(tsdiff_samples, axis=0)

    x_forecast = np.arange(split, split + prediction_length)

    # 3. Overlay the Predicted Continuation
    arima_predictions_idx = arima_predictions[idx]
    plt.figure(figsize=(10, 5))
    
    # Plot the full ground truth trajectory
    print(f'double_well_time[:split] shape: {double_well_time[:split].shape}, double_well_data[0, :split] shape: {double_well_data[0, :split].shape}')
    print(f'double_well_data[0, :split] shape: {double_well_data[0, :split].shape}, double_well_time[:split] shape: {double_well_time[:split].shape}')
    #plt.plot(double_well_time[:split], double_well_data[0, :split], label='Train', color='blue')
    # Plot ground truth for the first simulation
    plt.plot(x_forecast, double_well_data[0, split:split+prediction_length], label='True Test', color='green')

    # Plot ARIMA predictions
    plt.plot(x_forecast, arima_predictions_idx, color='blue', label='ARIMA', linewidth=2, linestyle='--')
    # Plot TSDiff median
    plt.plot(x_forecast, tsdiff_median, color='red', label='TSDiff Prediction', linewidth=2)


    # 4. Optional: Shade the uncertainty (Confidence Interval)
    '''plt.fill_between(x_forecast, 
                     np.percentile(tsdiff_samples, 5, axis=0), 
                     np.percentile(tsdiff_samples, 95, axis=0), 
                     color='red', alpha=0.2, label='TSDiff 90% CI')'''

    # Formatting
    #plt.axvline(x=split, color='gray', linestyle='--', alpha=0.5, label='Forecast Start')
    plt.title(f"TSDiff Unconditional Diffusion: True Series vs Predicted Continuation")
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.legend()
    #plt.grid(True, alpha=0.3)
    plt.savefig("./TESTING.png")

    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #---------------------------------# TSFDIFF Metrics #---------------------------------#
    #tsdiff_future_gt = all_targets[:, -prediction_length:]  # Ground truth future
    tsdiff_future_gt = all_targets[:, -prediction_length:].squeeze() 

    tsdiff_low90 = np.percentile(all_forecast_samples, 5, axis=1)
    tsdiff_high90 = np.percentile(all_forecast_samples, 95, axis=1)
    tsdiff_low50 = np.percentile(all_forecast_samples, 25, axis=1)
    tsdiff_high50 = np.percentile(all_forecast_samples, 75, axis=1)

    tsdiff_ci90_time = ((tsdiff_future_gt >= tsdiff_low90) & (tsdiff_future_gt <= tsdiff_high90)).mean(axis=0)
    tsdiff_ci50_time = ((tsdiff_future_gt >= tsdiff_low50) & (tsdiff_future_gt <= tsdiff_high50)).mean(axis=0)

    idx = 0
    gt_single = tsdiff_future_gt[idx]
    samples_single = all_forecast_samples[idx]

    tsdiff_mae_time = metrics.median_absolute_error_time(gt_single.reshape(1,-1), samples_single)
    all_tsdiff_crps = []
    for i in range(len(all_forecast_samples)):
        crps_i = metrics.crps_per_time(
            tsdiff_future_gt[i], 
            all_forecast_samples[i].T  # (T, N)
        )
        all_tsdiff_crps.append(crps_i)

    tsdiff_crps_time = torch.stack(all_tsdiff_crps).mean(dim=0)

    #tsdiff_mae_time = np.abs(tsdiff_future_gt - tsdiff_samples.mean(axis=1)).mean(axis=0)


    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------# plot metrics #-----------------------------------------#
    #-----------------------------------------#

    '''fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(18, 4)) 
    fig.suptitle("Model Comparison: ARIMA vs TSDiff")

    axes[0].plot(time_steps_test,average_coverage_90.detach().cpu().numpy(), label="ARIMA", color='blue')
    axes[0].plot(time_steps_test, tsdiff_ci90_time, label="TSDiff", color='green') # Added
    axes[0].axhline(0.9, color='black', linestyle='--')
    axes[0].set_title("Fraction inside 90% CI")
    axes[0].legend()

    axes[1].plot(time_steps_test, average_coverage_50.detach().cpu().numpy(), label="ARIMA", color='blue')
    axes[1].plot(time_steps_test, tsdiff_ci50_time, label="TSDiff", color='green') # Added
    axes[1].axhline(0.5, color='black', linestyle='--')
    axes[1].set_title("Fraction inside 50% CI")
    axes[1].legend()

    axes[2].plot(time_steps_test, arima_mae_time, color='blue', label="ARIMA")
    axes[2].plot(time_steps_test, tsdiff_mae_time, color='green', label="TSDiff") # Added
    axes[2].set_title("MAE (per time step)")
    axes[2].legend()

    axes[3].plot(time_steps_test, average_crps_time.detach().cpu().numpy(), color='blue', label="ARIMA tm")
    axes[3].plot(time_steps_test, average_crps_time_ps.detach().cpu().numpy(), color='blue',linestyle="--", label="ARIMA ps")
    axes[3].plot(time_steps_test, tsdiff_crps_time.detach().cpu().numpy(), color='green', label="TSDiff") # Uncomment if CRPS calculated
    axes[3].set_title("CRPS")
    axes[3].legend()
    plt.savefig("./double_well_results.png")
    plt.tight_layout()
    plt.show()'''



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="./results")
    
    args, _ = parser.parse_known_args()
    # Merge CLI overrides into config
    args = parser.parse_args()

    main(args.out_dir)