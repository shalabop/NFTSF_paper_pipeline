import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MDTrajectoryDataset(Dataset):
    def __init__(self, contexts, forecasts):
        assert len(contexts) == len(forecasts)

        self.context_length    = contexts.shape[1]
        self.prediction_length = forecasts.shape[1]

        ctx  = contexts.astype(np.float32)
        fcst = forecasts.astype(np.float32)

        self.contexts  = ctx
        self.forecasts = fcst

    def __len__(self):
        return len(self.contexts)
    
    def _create_positional_features(self, times):
        """
        Positional features for synthetic data.
        Designed to match the [-0.5, 0.5] range of the original
        calendar-based time features in Dataset_MTS.
        
        Four features, all in [-0.5, 0.5] or [-1, 1]:
        0: linear position normalized to [-0.5, 0.5]
        1: sin(2*pi*t)  — period = L+H steps
        2: cos(2*pi*t)  — period = L+H steps  
        3: sin(4*pi*t)  — period = (L+H)/2 steps (captures finer structure)
        """
        n = len(times)
        max_time = float(self.context_length + self.prediction_length)
        t_norm = times / max_time   # [0, 1)
        
        features = np.stack([
            t_norm - 0.5,                    # linear, [-0.5, 0.5)
            np.sin(2 * np.pi * t_norm),      # [-1, 1]
            np.cos(2 * np.pi * t_norm),      # [-1, 1]
            np.sin(4 * np.pi * t_norm),      # [-1, 1] finer period
        ], axis=1).astype(np.float32)        # (n, 4)
        
        return features

    def __getitem__(self, index):
        ctx  = self.contexts[index]   # (context_length,)
        fcst = self.forecasts[index]  # (prediction_length,)

        local_scaler = np.array([1.0], dtype=np.float32)

        batch_x = ctx[:, None]   
        batch_y = fcst[:, None]  
        
        x_mark = self._create_positional_features(times=
        np.arange(self.context_length)
        )
        y_mark = self._create_positional_features(times=
            np.arange(self.context_length, 
                    self.context_length + self.prediction_length)
        )
        
        return (
            torch.from_numpy(batch_x),
            torch.from_numpy(batch_y),
            torch.from_numpy(local_scaler),
            torch.from_numpy(x_mark),
            torch.from_numpy(y_mark),
            
        )


def _make_windows(positions, context_length, prediction_length, stride, T):
    """
    Slide windows with given stride over positions[:, :T].
    Returns contexts (N, ctx_len) and forecasts (N, pred_len).
    """
    seq_len   = context_length + prediction_length
    ctx_list  = []
    fcst_list = []
    for start in range(0, T - seq_len + 1, stride):
        ctx_list.append(positions[:, start:start + context_length])
        fcst_list.append(positions[:, start + context_length:start + seq_len])
    return (
        np.concatenate(ctx_list,  axis=0),   # (N, context_length)
        np.concatenate(fcst_list, axis=0),   # (N, prediction_length)
    )


def get_dataloader_md(npz_path, context_length, prediction_length,
                      batch_size, val_size, test_size,
                      stride, flag="train",train_test_split=None):
    data = np.load(npz_path)
    positions = data["positions"]
    
    if flag == "train":
        train_val_split = int(data["train_val_split"])
        val_start = train_val_split - context_length
            
        print(positions[:, :val_start].shape[0])
        print(positions[:, :val_start].shape[1])
        
        train_ctx, train_forecasts= _make_windows( positions[:,:], context_length, prediction_length,
        stride=stride, T=train_val_split,)
        
        train_dataset =  MDTrajectoryDataset(train_ctx, train_forecasts
        )
        
        val_ctx  = positions[:val_size, val_start:val_start + context_length]
        val_fcst = positions[:val_size, train_val_split:train_val_split+prediction_length]
        
        val_dataset = MDTrajectoryDataset(val_ctx,val_fcst)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=1)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=1)
        
        return train_loader, val_loader
    
    if flag == "test":
        print(f'train_test_split {train_test_split}')
        print(f'context_length {context_length}')
        print(f'prediction_length {prediction_length}')
        test_start = train_test_split - context_length
        test_end =  train_test_split + prediction_length
        
        test_ctx  = positions[:test_size, test_start:train_test_split]
        test_fcst = positions[:test_size, train_test_split:test_end]

        test_dataset = MDTrajectoryDataset(
            test_ctx, test_fcst
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=False, num_workers=1)
        return test_loader