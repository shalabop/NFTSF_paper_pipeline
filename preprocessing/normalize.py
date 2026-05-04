import argparse
import numpy as np
from pathlib import Path

##for data set where training and testing are fused

def normalize_all(datatype, out, data_path, train_test_split=None, train_val_split=None, train_size=None, time_steps=None):
    out_dir_unnorm = Path("data_unnorm") ##change this if you need the folder to be named something else
    out_dir_unnorm.mkdir(exist_ok=True)

    ##raw data would have shape (N,T)
    raw = np.load(data_path)
    if time_steps is not None:
        train_val_raw=raw[:train_size,:time_steps]
        test_raw = raw[train_size:,:time_steps]
    else:
        train_val_raw=raw[:train_size,:]
        test_raw = raw[train_size:,:]
    
    
    if train_val_split is None:
        train_val_split = int(0.9 * train_val_raw.shape[1])
    if train_test_split is None:
        train_test_split = int(0.9 * test_raw.shape[1])
    if datatype is None:
        datatype = Path(data_path).stem 
    train_val_data = train_val_raw
    test_data = test_raw    
    
    print(f'ds training  {train_val_data.shape}')
    print(f'ds testing  {test_data.shape}')
    
    print(f'saving raw data to {out_dir_unnorm}')
    
    
    print(f"train_val_split : {train_val_split}" )
    print(f"train_test_split : {train_test_split}" )
    
    mean = np.mean(train_val_data[:,:train_val_split])
    std = np.std(train_val_data[:,:train_val_split])
    print(f"Mean: {mean}, Std: {std}")
    
    time = np.arange(train_val_data.shape[1]) ##time is the same for train and val, and test
    
    ##for training all methods except tsdiff/tsdiff_cond, we only need the train_val.npz file, which includes the training and validation data normalized by the training set statistics. 
    # For tsdiff/tsdiff_cond, we will also create the gluonts_datasets with the same normalized data.
    np.savez(out_dir_unnorm/f"{datatype}_train.npz", positions=train_val_data, train_val_split=train_val_split, time=time,  mean=mean, std=std) 
    np.savez(out_dir_unnorm/f"{datatype}_test.npz", positions=test_data, train_test_split=train_test_split, time=time, mean=mean, std=std)
    np.savez(out_dir_unnorm/f"{datatype}.npz", positions_train=train_val_data, positions_test=test_data, train_test_split=train_test_split, time=time, mean=mean, std=std) ##for tsdiff/tsdiff_cond pipeline only
    
    ##create normlaized data using stats from training set only
    train_val_data_normalized = (train_val_data - mean) / (std + 1e-8)
    test_data_normalized = (test_data - mean) / (std + 1e-8)
    
    out_dir_norm=Path(out) 
    out_dir_norm.mkdir(exist_ok=True)
    
    np.savez(out_dir_norm/f"{datatype}_train.npz", positions=train_val_data_normalized, train_val_split=train_val_split, time=time, mean=mean, std=std)
    np.savez(out_dir_norm/f"{datatype}_test.npz", positions=test_data_normalized, train_test_split=train_test_split, time=time, mean=mean, std=std)
    np.savez(out_dir_norm/f"{datatype}.npz", positions_train=train_val_data_normalized, positions_test=test_data_normalized, train_test_split=train_test_split, time=time, mean=mean, std=std) ##for tsdiff/tsdiff_cond pipeline only
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Check reference quality")
    parser.add_argument("--data_path", "-d", type=str, required=True)
    parser.add_argument("--out_path", "-o", type=str, required=False, default="DATA")
    parser.add_argument("--train_test_split", "-tts", type=int, required=False)
    parser.add_argument("--train_val_split", "-tvs", type=int, required=False)
    parser.add_argument("--train_size", type=int,required=True) ##number of trajectories allocated for train/val, the rest will be for test
    parser.add_argument("--time_steps","-ts",type=int, required=False)
    parser.add_argument("--data_type","-dt", type=str, required=False) ##dataset name, used for naming the output files
    args = parser.parse_args()
    normalize_all(time_steps=args.time_steps,datatype=args.data_type,out=args.out_path, data_path=args.data_path, train_test_split=args.train_test_split, train_val_split=args.train_val_split, train_size=args.train_size)
    print("==="*20)
