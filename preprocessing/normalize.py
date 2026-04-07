### normalize the data using Z-score to the following format
# data_train.npz (includes train + val sets normalized by the training set statistics only) includes:
####   - positions (N ,T) (train + val)
####   - train_val_split (the index at which validation target starts, midway between prediction target in validation and observed past at the end of the training) 
####   - time (T)
####   - prediction_length
####   - context_length
# data_test.npz (includes the test data set normalized with the training data statistics)
#### - positions (N, T) (testing only set)
#### - train_test_split (the index at which forecasting starts)
#### - time (T)
#### - preidiction_length
#### - context_length 
# data.npz (file only for creating gluontes_datasets for tsdiff/tsdiff_cond pipeline only)
#### creates folder containting train and test subfolders with temporal split between training and validation

##automatically creates the files above for the following datasets
# double_well
# single_well
# alanine_phi
# alanine_psi
# linear_gaussian


##takes unnormalized data from ./DATA_unnorm and creates DATA and gluonts_datasets where the normalized data is at now 
import numpy as np
from pathlib import Path

def normalize_all(prediction_length=100, context_length=100, train_test_split=900, train_val_split=900):
    ##must be named exactly
    datasets = ["double_well","single_well", "alanine_phi", "alanine_psi", "linear_gaussian"]
    out_dir = Path("DATA_unnorm") ##change this if you need the folder to be named something else
    out_dir.mkdir(exist_ok=True)
    

    for ds in datasets:
        print(f"\n=================================================================")
        print(f"Processing {ds}")
        print(f"=================================================================")

        #DATA_unnorm where is the raw unnormalized is 
        train_val_raw = Path(f"DATA_raw/{ds}_train.npy")
        test_raw = Path(f"DATA_raw/{ds}_test.npy")

        if not train_val_raw.exists() or not test_raw.exists():
            print(f"  Missing raw files for {ds}, skipping.")
            continue
        
        train_val_data = np.load(train_val_raw)
        test_data = np.load(test_raw)
        print(train_val_data[:5,0]) ##starting from one
        print(test_data[:5,0]) ##starting from one
        
        
        
        print(f'ds training  {train_val_data.shape}')
        print(f'ds testing  {test_data.shape}')
        
        print("==="*15)
        print("AFTER removing index coloumn")
        print("==="*15)
        
        ###take the first row out (containing indices)
        train_val_data_processed = train_val_data[:,1:] ##starting from one
        test_data_processed = test_data[:,1:]##starting from one
        print(train_val_data_processed[:5,0])
        print(test_data_processed[:5,0])
        
        print(f'ds training  {train_val_data_processed.shape}')
        print(f'ds testing  {test_data_processed.shape}')
        
        print("==="*15)
        print("AFTER transposing")
        print("==="*15)
        
        train_val_data_transposed = train_val_data_processed.T
        test_data_transposed = test_data_processed.T

        print(f'ds training  {train_val_data_transposed.shape}')
        print(f'ds testing  {test_data_transposed.shape}')
        
        train_data, val_data = train_val_data_transposed[:,:train_val_split],train_val_data_transposed[:,train_val_split:]
        print("==="*15)
        print("TRAIN and VAL")
        print("==="*15)
        print(f'Training set: {train_data.shape}')
        print(f'Validation set: {val_data.shape}')
        
        
        print("==="*15)
        print("Calculating mean and std from training set only")
        print("==="*15)
        
        mean = np.mean(train_data)
        std = np.std(train_data)
        print(f"Mean: {mean}, Std: {std}")
        
        time = np.arange(train_val_data_transposed.shape[1]) ##time is the same for train and val, and test
        
        ##for training all methods except tsdiff/tsdiff_cond, we only need the train_val.npz file, which includes the training and validation data normalized by the training set statistics. 
        # For tsdiff/tsdiff_cond, we will also create the gluonts_datasets with the same normalized data.
        np.savez(out_dir/f"{ds}_train.npz", positions=train_val_data_transposed, train_val_split=train_val_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std) 
        np.savez(out_dir/f"{ds}_test.npz", positions=test_data_transposed, train_test_split=train_test_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std)
        np.savez(out_dir/f"{ds}.npz", positions_train=train_val_data_transposed, positions_test=test_data_transposed, train_test_split=train_test_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std) ##for tsdiff/tsdiff_cond pipeline only
        
        ##create normlaized data using stats from training set only
        train_val_data_normalized = (train_val_data_transposed - mean) / (std + 1e-8)
        test_data_normalized = (test_data_transposed - mean) / (std + 1e-8)
        
        ##save the normalized data in the same format as above for tsdiff/tsdiff_cond pipeline only
        out_dir_norm=Path("DATA") ##change this if you need the folder to be named something else
        out_dir_norm.mkdir(exist_ok=True)
        
        
        np.savez(out_dir_norm/f"{ds}_train.npz", positions=train_val_data_normalized, train_val_split=train_val_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std)
        np.savez(out_dir_norm/f"{ds}_test.npz", positions=test_data_normalized, train_test_split=train_test_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std)
        np.savez(out_dir_norm/f"{ds}.npz", positions_train=train_val_data_normalized, positions_test=test_data_normalized, train_test_split=train_test_split, time=time, prediction_length=prediction_length, context_length=context_length, mean=mean, std=std) ##for tsdiff/tsdiff_cond pipeline only



if __name__ == '__main__':
    normalize_all()
    print("==="*20)
