import numpy as np
from torch.utils.data import DataLoader, Dataset


class MDTrajectoryDataset(Dataset):
    def __init__(self, positions, context_length, prediction_length,
                 normalization):
        
        """
        positions     : (N, seq_len) — raw un-normalized windows
        mean          : scalar — global training mean
        std           : scalar — global training std
        normalization : "zscore" | "std" | "local" | "none"

        For "local": no global normalization applied here.
                     Per-sample scaler computed in __getitem__ from context.
                     Scaler returned in batch as "local_scaler" (B, 1).
        """
        assert normalization in ("zscore", "std", "local", "none"), \
            f"normalization must be 'zscore', 'std', 'local', or 'none', " \
            f"got '{normalization}'"

        self.context_length    = context_length
        self.prediction_length = prediction_length
        self.seq_len           = context_length + prediction_length
        self.normalization     = normalization
        N                      = len(positions)

        pos = positions.astype(np.float32)
        '''if normalization == "zscore":
            pos = (pos - mean) / std
        elif normalization == "std":
            pos = pos / std'''

        self.positions = pos

        self.observed_mask = np.ones(
            (N, self.seq_len, 1), dtype=np.float32)
        self.gt_mask = np.ones(
            (N, self.seq_len, 1), dtype=np.float32)
        self.gt_mask[:, -self.prediction_length:, :] = 0.0

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        seq = self.positions[index, :self.seq_len][:, None]  # (L, 1)

        '''if self.normalization == "local":
            ctx    = seq[:self.context_length, 0]          
            scaler = float(np.abs(ctx).mean())
            scaler = max(scaler, 1e-8)                     
            seq    = (seq / scaler).astype(np.float32)
            local_scaler = np.array([scaler], dtype=np.float32)
        else:
            local_scaler = np.array([1.0], dtype=np.float32)  '''
        local_scaler = np.array([1.0], dtype=np.float32)
        return {
            "observed_data" : seq,
            "observed_mask" : self.observed_mask[index],
            "gt_mask"       : self.gt_mask[index],
            "timepoints"    : np.arange(self.seq_len, dtype=np.float32),
            "feature_id"    : np.array([0], dtype=np.float32),
            "local_scaler"  : local_scaler,  
        }


def _make_train_windows(positions, context_length, prediction_length, stride):
    seq_len = context_length + prediction_length
    T       = positions.shape[1]
    windows = []
    for start in range(0, T - seq_len + 1, stride):
        windows.append(positions[:, start:start + seq_len])
    return np.concatenate(windows, axis=0)


def get_dataloader_md(npz_path, context_length, prediction_length,
                      batch_size, val_size, test_size,
                      stride, normalization,flag="train"):
    data = np.load(npz_path)
    positions = data["positions"]

    print(f"Number of trajectories: {positions.shape[0]}")
    print(f"Number of time steps: {positions.shape[1]}")


    '''mean = float(positions[:, :train_end].mean())
    std  = float(positions[:, :train_end].std())

    print(f"normalization : {normalization}")
    print(f"mean          : {mean:.4f}  std: {std:.4f}")
    if normalization == "local":
        print("  (local: per-sample scaler = mean(|context|), ""stored in batch as 'local_scaler')")'''

    if flag == "train":
        val_start = -((prediction_length + context_length) - positions.shape[1])
        
        print(positions[:, :val_start].shape[0])
        print(positions[:, :val_start].shape[1])
        train_dataset = MDTrajectoryDataset(
            _make_train_windows(positions[:, :val_start],context_length, prediction_length, stride),
            context_length, prediction_length, normalization=normalization,
        )
        
        print(positions[:val_size, val_start:].shape[0])
        print(positions[:val_size, val_start:].shape[1])
        val_dataset = MDTrajectoryDataset(positions[:val_size, val_start:],context_length, prediction_length, normalization=normalization,
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,  num_workers=1)
        
        val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False, num_workers=1)
    
        return train_loader, val_loader
    
    elif flag == "test":
        print(positions[:test_size,-(prediction_length+context_length):].shape)
        test_dataset = MDTrajectoryDataset(
            positions[:test_size,-(prediction_length+context_length):],
            context_length, prediction_length, normalization=normalization,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=1)
        return test_loader