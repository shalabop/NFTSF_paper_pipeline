import argparse
from matplotlib import pyplot as plt
import numpy as np
import torch
from pathlib import Path
import logging
import torch
import pandas as pd
from pathlib import Path
from gluonts.dataset.jsonl import JsonLinesWriter
from gluonts.dataset.common import TrainDatasets, MetaData,FileDataset
import yaml
import pandas as pd

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def create_gluonts_dataset(
    dataset_path: Path,
    positions_train: torch.Tensor,
    positions_test: torch.Tensor,
    #train_test_split: int,
    #prediction_length: int,
    pandas_freq: str = "H",
):
    """
    Converts a tensor of trajectories into a GluonTS-compatible dataset
    (train/test JSON files) using a fixed time-step index as the split.

    Parameters
    ----------
    dataset_path     : directory to write the dataset into
    positions        : (N, T) tensor or numpy array of trajectories
    train_test_split : time-step index separating train and test
    prediction_length: number of future steps to forecast
    pandas_freq      : pandas frequency string (default "H", used as a label)
    test_dataset_size : int to denote the number of trajectories to include in the test dataset 
    """
    if isinstance(dataset_path, str):
        dataset_path = Path(dataset_path)

    dataset_path.mkdir(parents=True, exist_ok=True)

    number_of_trajectories_train, total_steps_train = positions_train.shape
    number_of_trajectories_test, total_steps_test = positions_test.shape

    '''if train_test_split >= total_steps:
        raise ValueError(
            f"train_test_split ({train_test_split}) must be less than "
            f"total_steps ({total_steps})."
        )
    if train_test_split + prediction_length > total_steps:
        raise ValueError(
            f"train_test_split ({train_test_split}) + prediction_length "
            f"({prediction_length}) = {train_test_split + prediction_length} "
            f"exceeds total_steps ({total_steps})."
        )
    if train_test_split + prediction_length != total_steps:
        print("WARNING: gluonTS would assume the prediction length \
              ends at the end of the dataset")'''

    train_data, test_data = [], []
    start_time = pd.Timestamp("2026-01-01 00:00:00")

    for i in range(number_of_trajectories_train):
        if isinstance(positions_train, torch.Tensor):
            trajectory = positions_train[i].detach().cpu().tolist()
        else:
            trajectory = positions_train[i].tolist()

        '''train_data.append({
            "start" : start_time,
            "target": trajectory[:train_test_split],
            "item_id": i,
        })'''
        
        train_data.append({
            "start" : start_time,
            "target": trajectory,
            "item_id": i,
        })

        #only include in the test dataset up to test_dataset_size examples
        '''if i < test_dataset_size:
            test_data.append({
                "start" : start_time,
                "target": trajectory,   
                "item_id": i,  
            })'''
        
    for i in range(number_of_trajectories_test):
        if isinstance(positions_test, torch.Tensor):
            trajectory = positions_test[i].detach().cpu().tolist()
        else:
            trajectory = positions_test[i].tolist()

        test_data.append({
            "start" : start_time,
            "target": trajectory,   
            "item_id": i,  
        })

    meta = MetaData(
        freq=pandas_freq,
        prediction_length=100,
        feat_static_cat=[{
            "name"       : "item_id",
            "cardinality": str(number_of_trajectories_test),
        }],
    )

    writer  = JsonLinesWriter(use_gzip=False, suffix=".json")
    dataset = TrainDatasets(metadata=meta, train=train_data, test=test_data)
    dataset.save(str(dataset_path), writer=writer, overwrite=True)

    print(f"Dataset saved to : {dataset_path}")
    print(f"  train series   : {len(train_data)}")
    print(f"  test  series   : {len(test_data)}")
    #print(f"  train length   : {train_test_split} steps")
    #print(f"  prediction len : {prediction_length} steps")

def read_data(input_path: Path) -> dict:
    """
    Loads a .npz data file and returns a config dict.
    CLI overrides take priority over values stored in the .npz.

    Parameters
    ----------
    input_path        : path to .npz file produced by generator.py
    split_override    : CLI --split value, or None to use stored value
    pred_len_override : CLI --prediction-length value, or None to use stored value

    Returns
    -------
    cfg : dict with keys:
        positions         (N, T) numpy array
        time              (T,)   numpy array
        train_test_split  int
        prediction_length int
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.suffix != ".npz":
        raise ValueError(f"Expected a .npz file, got: {input_path.suffix}")

    data = np.load(input_path, allow_pickle=False)

    '''if "positions" not in data:
        raise KeyError(f"'positions' array not found in {input_path}")
    if "time" not in data:
        raise KeyError(f"'time' array not found in {input_path}")'''

    positions_train = data["positions_train"]   
    positions_test = data["positions_test"]
    #time      = data["time"]
    time      = np.arange(positions_train.shape[1])       

    '''if "train_test_split" in data:
        train_test_split = int(data["train_test_split"])
        logger.info(f"  train_test_split  : {train_test_split} (from .npz)")
    else:
        raise ValueError(
            "ERORR: train_test_split not found in .npz"
        )'''

    '''if "prediction_length" in data:
        prediction_length = int(data["prediction_length"])
        logger.info(f"  prediction_length : {prediction_length} (from .npz)")
    else:
        raise ValueError(
            "ERORR: prediction_length not found in .npz"
        )'''

    #N, T =  .shape
    '''if train_test_split >= T:
        raise ValueError(
            f"train_test_split ({train_test_split}) must be less than "
            f"total time steps ({T})."
        )
    if train_test_split + prediction_length > T:
        raise ValueError(
            f"train_test_split ({train_test_split}) + prediction_length "
            f"({prediction_length}) = {train_test_split + prediction_length} "
            f"exceeds total time steps ({T})."
        )'''

    return {
        "positions_train"        : positions_train,
        "positions_test"         : positions_test,
        "time"                   : time,
        #"train_test_split" : train_test_split,
        #"prediction_length": prediction_length,
    }

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a .npz data file (from generator.py) into a "
            "GluonTS-compatible dataset (train/test JSON files)."
        )
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input .npz file (e.g. data/double_well.npz)",
    )
    parser.add_argument(
        "--out-dir", "-o",
        required=True,
        help="Output directory for GluonTS dataset (e.g. datasets/double_well)",
    )

    parser.add_argument(
        "--test-size","-ts",
        required=False,
        type=int,
        default=0,
        help="The Numbee of Time series to include in the Test Dataset",
    )

    parser.add_argument(
        "--testing","-t",
        type=bool,
        default=True,
        required=False,
        help=(
            "Run testing scripts by plotting the data and running checks for the data type "
        )
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)

    logger.info(f"Reading input: {input_path}")
    cfg = read_data(input_path)

    positions_train         = cfg["positions_train"]           
    positions_test          = cfg["positions_test"]
    time =cfg["time"]
    #train_test_split  = cfg["train_test_split"]
    #prediction_length = cfg["prediction_length"]


    N, T = positions_train.shape
    logger.info(f"  num trajectories  : {N}")
    logger.info(f"  total time steps  : {T}")
    #logger.info(f"  train steps       : {train_test_split}")
    #logger.info(f"  prediction length : {prediction_length}")

    positions_train = torch.tensor(positions_train, dtype=torch.float32)
    positions_test = torch.tensor(positions_test, dtype=torch.float32)

    logger.info(f"Writing GluonTS dataset to: {out_dir}")
    test_dataset_size=args.test_size
    create_gluonts_dataset(
        dataset_path    = out_dir,
        positions_train = positions_train,
        positions_test = positions_test,
        #time             = time
    )

    print(f'--------------------------------{out_dir / "time.npz"}--------------------------------')
    np.savez_compressed(
        out_dir / "time.npz", #save time inside the die as .npz
        time = time,
        train_test_split = 800,
        time_test = cfg["time"][800:],
        time_train = cfg["time"][:]
    )

    if args.testing:
        dataset_path=Path(f"{out_dir}")
        with open(dataset_path/"metadata.json", "r") as f:
            meta_json = yaml.safe_load(f)
        metadata = MetaData(freq=meta_json["freq"], prediction_length=meta_json["prediction_length"])
        train_ds = FileDataset(dataset_path / "train", freq=metadata.freq)
        test_ds = FileDataset(dataset_path / "test", freq=metadata.freq)
        dataset = TrainDatasets(metadata=metadata, train=train_ds, test=test_ds)

        test_list = list(dataset.test)
        train_list = list(dataset.train)
        logger.info("RUNNING Test #1: Size of the Test Dataset")
        gluonts_t_dict=np.load(dataset_path/"time.npz")
        gluonts_time=gluonts_t_dict["time"]
        gluonts_split=int(gluonts_t_dict["train_test_split"])
        
        '''logger.info(f'GluonTS ---------------------------  VS --------------------------- Raw .npz')
        logger.info(f'Test #1 Train dataset empty? {len(test_list)==0} --------------------------- {len(positions)==0}')
        gluonts_x_train = torch.stack([torch.tensor(item["target"]) for item in train_list])
        logger.info(f'Test #1 Train dataset shape ? {gluonts_x_train.shape} --------------------------- {positions[:,:train_test_split].shape}')    
        logger.info(f'Test #1 Train dataset time steps ? {gluonts_x_train.shape[1]} --------------------------- {positions[:,:train_test_split].shape[1]}')
        logger.info(f'Test #1 Train dataset trajectories ? {gluonts_x_train.shape[0]} --------------------------- {positions[:,:train_test_split].shape[0]}')


        logger.info(f'Test #1 is Test dataset empty? {len(test_list)==0}')
        gluonts_x_test = torch.stack([torch.tensor(item["target"]) for item in test_list])
        logger.info(f'Test #1 Test dataset shape ? {gluonts_x_test.shape}')    
        logger.info(f'Test #1 Test dataset time steps ? {gluonts_x_test.shape[1]}')
        logger.info(f'Test #1 Test dataset trajectories ? {gluonts_x_test.shape[0]}')
        logger.info(f'Test #1 Split At ? {gluonts_split}')
        logger.info(f'Test #1 Total Time Steps ? {gluonts_time.shape}')
        logger.info(f'Test #1 Test Time Steps ? {gluonts_time[gluonts_split:].shape}')
        logger.info("FINISHED Test #1: Size of the Test Dataset")

        ##confirms writing the correct data
        logger.info("RUNNING Test #2: plotting what is being written vs after writing")
        fig,axes=plt.subplots(ncols=3,nrows=1,figsize=(10,4))
        for i in range(6):
            axes[0].plot(time[:train_test_split],positions[i][:train_test_split], color='blue')
            axes[0].plot(time[train_test_split:],positions[i][train_test_split:], color='red')

        for i in range(gluonts_x_test.shape[0]):
            axes[1].plot(gluonts_time[:gluonts_split],gluonts_x_test[i][:gluonts_split], color='blue')
            axes[1].plot(gluonts_time[gluonts_split:],gluonts_x_test[i][gluonts_split:], color='red')

        for i in range(gluonts_x_train.shape[0]):
            axes[2].plot(gluonts_time[:gluonts_split],gluonts_x_train[i], color='blue')

        axes[0].set_title(".npz data")
        axes[1].set_title("gluonTS Test Dataset")
        axes[2].set_title("gluonTS Train Dataset")
        #plt.savefig("test_plots/Test #2: plotting what is being written vs after writing")
        logger.info(f"Test #2 gluonTS test data == raw .npz shape? {positions.shape==gluonts_x_test.shape}")
        logger.info(f"Test #2 gluonTS test data & raw .npz types? {type(positions),type(gluonts_x_test)}")
        logger.info(f"Test #2 gluonTS test data == raw .npz data? {np.all(positions.numpy()==gluonts_x_test.numpy())}")

        train_list = list(dataset.train)
        gluonts_x_train = torch.stack([torch.tensor(item["target"]) for item in train_list])

        logger.info(f"Test #2 gluonTS train data == raw .npz shape? {positions[:,:train_test_split].shape==gluonts_x_train.shape}")
        logger.info(f"Test #2 gluonTS train data & raw .npz types? {type(positions[:,:train_test_split]),type(gluonts_x_train)}")
        logger.info(f"Test #2 gluonTS train data == raw .npz data? {np.all(positions[:,:train_test_split].numpy()==gluonts_x_train.numpy())}")
        logger.info("FINISHED Test #2: plotting what is being written vs after writing")'''
    
    logger.info("Done.")
    logger.info(f"  Train : {out_dir}/train/")
    logger.info(f"  Test  : {out_dir}/test/")
    logger.info(f"  Meta  : {out_dir}/metadata.json")


if __name__ == "__main__":
    main()