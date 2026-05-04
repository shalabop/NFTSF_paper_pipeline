import argparse
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pmdarima as pm
from pathlib import Path
import logging
from tqdm.auto import tqdm
import yaml

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def plot_forecast(time, full_trajectory, split, forecasts,results_dw,**kwargs):
    ci90_lower=kwargs.get("ci90_lower")
    ci90_upper=kwargs.get("ci90_upper")
    ci50_upper=kwargs.get("ci50_upper")
    ci50_lower=kwargs.get("ci50_lower")
    filename = kwargs.get("filename")

    fig,axes=plt.subplots(nrows=4,ncols=2,figsize=(12,12))
    
    for i in range(8):
        axes[i//2,i%2].plot(time,full_trajectory[i])
        axes[i//2,i%2].plot(time[split:],forecasts[i].predicted_mean) 
        sims = results_dw[j].simulate(
            nsimulations=len(time[split:]), 
            repetitions=1,
            anchor='end'
        )
        
        for col in sims.columns:
            if col==5:
                break
            axes[i//2,i%2].plot(time[split:], sims[col], color='blue', alpha=0.3) 
        
        #plot confidence interval 90ci
        axes[i//2,i%2].fill_between(time[split:], ci90_lower[i], ci90_upper[i], color='orange', alpha=0.3, label='90% CI')
        axes[i//2,i%2].fill_between(time[split:], ci50_lower[i], ci50_upper[i], color='green', alpha=0.3, label='50% CI')
        #denote hte wells   
        axes[i//2,i%2].axhline(y=1, color='red', linestyle='--', alpha=0.6, label='Left Well')
        axes[i//2,i%2].axhline(y=-1, color='purple', linestyle='--', alpha=0.6, label='Right Well')
        axes[i//2,i%2].set_xlabel("Time steps")
        axes[i//2,i%2].set_ylabel("Position")

    if not filename:
        date=filename = datetime.now().strftime("plot_%Y%m%d_%H%M%S.png")
        filename=f"./plots/ARIMA{date}"
    plt.savefig(filename)
    plt.tight_layout()

def read_data(input_path: Path, split_override: int | None, predection_length_override: int | None,) -> dict:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix != ".npz":
        raise ValueError(
            f"Expected a .npz file, got: '{input_path.suffix}'. "
            f"Make sure you're passing the output of generator.py."
        )

    data = np.load(input_path, allow_pickle=False)

    if "positions" not in data:
        raise KeyError(f"'positions' array not found in {input_path}")
    if "time" not in data:
        raise KeyError(f"'time' array not found in {input_path}")

    positions = data["positions"]   
    time = data["time"]  
    train_test_split = data.get("train_test_split")
    prediction_length = data.get("prediction_length")      

    if split_override is not None:
        train_test_split = int(split_override)
        logger.info(f"  train_test_split  : {train_test_split} (CLI override)")
    else:
        train_test_split = int(data["train_test_split"])
        logger.info(f"  train_test_split  : {train_test_split} (from .npz)")


    if predection_length_override is not None:
        prediction_length = int(predection_length_override)
        logger.info(f"prediction_length : {prediction_length} (CLI override)")
    elif "prediction_length" in data:
        prediction_length = int(data["prediction_length"])
        logger.info(f"prediction_length : {prediction_length} (from .npz)")
    else:
        raise ValueError(
            "prediction_length not found in .npz and --prediction-length not provided."
        )

    N, T = positions.shape
    if train_test_split >= T:
        raise ValueError(
            f"train_test_split ({train_test_split}) must be less than "
            f"total time steps ({T})."
        )
    if train_test_split + prediction_length > T:
        raise ValueError(
            f"train_test_split ({train_test_split}) + prediction_length "
            f"({prediction_length}) = {train_test_split + prediction_length} "
            f"exceeds total time steps ({T})."
        )

    return {
        "positions" : positions,
        "time" : time,
        "train_test_split" : train_test_split,
        "prediction_length": prediction_length,
    }

def fit_and_forecast(
    positions: np.ndarray,
    train_test_split: int,
    num_of_samples: int,
    use_validation_set: bool,
    label:str,
    prediction_length: int = None,
    validation_metric: str = "mse",
    save_figures: bool = False,
    time : np.array = None ,
    number_of_forecasts_to_plot : int =6,
    path_to_save_figures: str = "./plots/ARIMA",
    context_length: int = None
) -> dict:
    N = positions.shape[0]

    if not prediction_length:
        prediction_length=len(positions[train_test_split:])#the entire test set 
        
    print(prediction_length)
    print(num_of_samples)
    print(train_test_split)

    full_trajectories = positions
    samples = np.zeros((N, prediction_length, num_of_samples))
    ground_truth = positions[:, train_test_split : train_test_split + prediction_length]

    ci90_lower_est = np.zeros((N, prediction_length)) 
    ci90_upper_est = np.zeros((N, prediction_length)) #for each model, we will have a confidence interval for each time step in the prediction horizon
    ci50_lower_est = np.zeros((N, prediction_length))
    ci50_upper_est = np.zeros((N, prediction_length))

    fig,axes=plt.subplots(nrows=4,ncols=2,figsize=(12,12))

    if save_figures:
        date = datetime.datetime.now().strftime("ARIMA_forecasts_%Y%m%d_%H%M%S.png")
        filename=f"./plots/ARIMA/{label}_{date}"

    time_start = time.time()
    for i in tqdm(range(N), desc="Fitting ARIMA models"):
        #train_series = pd.Series(positions[i, :train_test_split])
        #np.random.seed(42)
        if context_length is not None:
            train_series = pd.Series(positions[i, train_test_split-context_length:train_test_split])
            #model  = pm.auto_arima(train_series, metric=validation_metric,random_state=42+i)
            model  = pm.auto_arima(train_series, random_state=42+i)
            print(f"Only using the past{context_length} for forecasting")
        else:
            train_series = pd.Series(positions[i, :train_test_split])
            print(f"Using the entire history of {positions.shape[1]-prediction_length} steps for forecasting")
            model  = pm.auto_arima(train_series, out_of_sample_size=prediction_length, metric=validation_metric,random_state=42+i)
        '''if use_validation_set:
            model  = pm.auto_arima(train_series, out_of_sample_size=prediction_length, metric=validation_metric)
        else:
            mode= pm.auto_arima(train_series)'''
       
        result = model.arima_res_
        #np.random.seed(42)
        #forecast    = result.get_forecast(steps=prediction_length)
        sims = result.simulate(nsimulations=prediction_length,repetitions=num_of_samples,anchor="end",random_state=42+i)
        
        #calculate ci bands from samples
        sims = np.asarray(sims).squeeze()
        samples[i] = sims   
        ci90_lower_est[i] = np.percentile(sims,5,axis=1)
        ci90_upper_est[i] = np.percentile(sims,95,axis=1)
        ci50_lower_est[i] = np.percentile(sims,25,axis=1)
        ci50_upper_est[i] = np.percentile(sims,75,axis=1)
        
    total_elapsed = time.time() - time_start

    ##only if time is passed 
    if save_figures:
        if time is None:
            logger.info("Plotting FAILED required parameter time missing")
        else:    
            plt.figure(figsize=(10, 4))
            plt.tight_layout()
            for i in range(6):
                plt.plot(time[:train_test_split+prediction_length],full_trajectories[i,:train_test_split+prediction_length],color="blue")

            for i in range(6):
                plt.plot(time[train_test_split:],full_trajectories[i,train_test_split:],color="green")
            plt.xlabel("Time steps")
            plt.ylabel("Position")
            plt.title(f"ARIMA {label}")
            plt.legend()
            if path_to_save_figures:
                Path(path_to_save_figures).mkdir(parents=True, exist_ok=True)
                filename = datetime.datetime.now().strftime(f"full_trajectories_{label}_%Y%m%d_%H%M%S.png")
            plt.savefig(f'{path_to_save_figures}/{filename}')

            ig,axes=plt.subplots(nrows=4,ncols=2,figsize=(12,12))
            plt.tight_layout(pad=2.0)

    return {
        "samples"     : samples,       
        "ground_truth": ground_truth,  
        "ci90_lower"  : ci90_lower_est,    
        "ci90_upper"  : ci90_upper_est,    
        "ci50_lower"  : ci50_lower_est,    
        "ci50_upper"  : ci50_upper_est,    
        'full_trajectories' : full_trajectories,
        "total_elapsed" : total_elapsed
    }

def main():
    parser = argparse.ArgumentParser(
        description="Fit one auto-ARIMA model per trajectory and save forecast bundle."
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to .npz data file from generator.py (e.g. data/double_well.npz)",
    )
    parser.add_argument(
        "--out", "-o",
        required=True,
        help="Output .npz path (e.g. predictions/arima_double_well.npz)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        required=False,
        default=1000,
        help="Number of Monte Carlo simulations per ARIMA model (default: 100)",
    )
    parser.add_argument(
        "--split",
        type=int,
        required=False,
        default=None,
        help="Train/test split index. Overrides value stored in .npz if provided.",
    )
    parser.add_argument(
        "--prediction-length",
        required=False,
        type=int,
        default=None,
        help="Forecast horizon. Overrides value stored in .npz if provided.",
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        default=None,
        required=False,
        help="Path to forecasting config file (e.g. tsfdiff_forecast_config/double_well.yaml)",
    )

    parser.add_argument(
        "--num-models", "--models",
        type=int,
        default=None,
        help="Number of ARIMA models to fit. Must be <= number of trajectories in the dataset. If not provided, fits one model per trajectory.",
    )

    parser.add_argument(
        "--label", "-l",
        type=str,
        default="Unknown",
        help="Label for the plots.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    out_path   = Path(args.out)

    logger.info(f"Reading input: {input_path}")
    input_data = read_data(input_path, args.split, args.prediction_length)

    positions = input_data["positions"]
    time = input_data["time"]
    train_test_split = input_data["train_test_split"]
    #print(train_test_split)
    #print(time)

    #prediction_length = input_data["prediction_length"]
    use_validation_set = False
    num_of_samples = args.num_samples
    print(num_of_samples)
#    exit()

    if args.config is not None:
        logger.info(f"Reading forecasting config from: {args.config}")
        with open(args.config, "r") as f:
            forecast_config = yaml.safe_load(f)
            prediction_length = forecast_config.get("prediction_length")
            context_length = forecast_config.get("context_length")
            use_validation_set= forecast_config.get("use_validation_set")
            #num_of_samples= forecast_config.get("num_samples")
            #train_test_split = forecast_config.get("train_test_split")

    N, T = positions.shape

    logger.info(f"Loaded data with {N} trajectories and {T} time steps.")
    logger.info(f"  trajectories      : {N}")
    logger.info(f"  total time steps  : {T}")
    logger.info(f"  num samples   : {num_of_samples}")

    N_models = args.num_models if args.num_models is not None else N
    print(f"Fitting {N_models} ARIMA models...")
    positions = positions[:N_models] 

    print(f"Prediction length: {prediction_length}")
    print(f"Using validation set: {use_validation_set}")
    print(f'Number of samples: {num_of_samples}')
    print(f'test set time steps: {len(time[train_test_split:train_test_split + prediction_length])}')
    print(f'train set time steps: {len(time[:train_test_split])}')
    print(f'train test split: {train_test_split}')
    print(f'data contains NaNs: {np.isnan(positions).any()}')

    results = fit_and_forecast(
        positions        = positions,
        train_test_split = train_test_split,
        prediction_length= prediction_length,
        num_of_samples  = num_of_samples,
        use_validation_set = use_validation_set,
        time = time,
        label=args.label,
        context_length=context_length
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix != ".npz":
        out_path = out_path.with_suffix(".npz")
        logger.warning(f"Changing output extension to .npz → {out_path}")

    
    np.savez(
        out_path,
        samples = results["samples"],        
        ground_truth = results["ground_truth"],    
        ci90_lower = results["ci90_lower"],      
        ci90_upper = results["ci90_upper"],      
        ci50_lower = results["ci50_lower"],     
        ci50_upper = results["ci50_upper"],     
        full_trajectories = results["full_trajectories"],
        time_test = time[train_test_split:train_test_split + prediction_length],
        time_train = time[:train_test_split],
        time = time,
        train_test_split = train_test_split,
        prediction_length = prediction_length,
        num_of_samples = num_of_samples,
    )

    logger.info(f"Saved forecast bundle to: {out_path}")
    logger.info(f"samples shape : {results['samples'].shape}")
    logger.info(f"ground_truth shape : {results['ground_truth'].shape}")

if __name__ == "__main__":
    main()