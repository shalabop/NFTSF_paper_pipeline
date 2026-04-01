import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

class MDCCDMDataset(Dataset):
    def __init__(self, contexts, forecasts, mean, std, normalization="none"):
        """
        contexts      : (N, context_length)    raw un-normalized
        forecasts     : (N, prediction_length) raw un-normalized
        mean          : scalar — global training mean
        std           : scalar — global training std
        normalization : "zscore" | "std" | "local" | "none"
        """
        assert normalization in ("zscore", "std", "local", "none"), \
            f"normalization must be 'zscore', 'std', 'local', or 'none', " \
            f"got '{normalization}'"
        assert len(contexts) == len(forecasts)

        self.context_length    = contexts.shape[1]
        self.prediction_length = forecasts.shape[1]
        self.normalization     = normalization
        self.mean = mean
        self.std  = std

        ctx  = contexts.astype(np.float32)
        fcst = forecasts.astype(np.float32)

        if normalization == "zscore":
            ctx  = (ctx  - mean) / std
            fcst = (fcst - mean) / std
        elif normalization == "std":
            ctx  = ctx  / std
            fcst = fcst / std

        self.contexts  = ctx
        self.forecasts = fcst

    def __len__(self):
        return len(self.contexts)
    
    def _create_positional_features(self, times):
        n = len(times)
        features = np.zeros((n, 4), dtype=np.float32)

        max_time = self.context_length + self.prediction_length
        t_norm = times / max_time

        features[:, 0] = t_norm
        features[:, 1] = np.sin(2 * np.pi * t_norm)
        features[:, 2] = np.cos(2 * np.pi * t_norm)
        features[:, 3] = t_norm ** 2

        return features

    def __getitem__(self, index):
        ctx  = self.contexts[index]   # (context_length,)
        fcst = self.forecasts[index]  # (prediction_length,)

        if self.normalization == "local":
            scaler = float(np.abs(ctx).mean())
            scaler = max(scaler, 1e-8)
            ctx    = (ctx  / scaler).astype(np.float32)
            fcst   = (fcst / scaler).astype(np.float32)
            local_scaler = np.array([scaler], dtype=np.float32)
        else:
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
        '''return (
            torch.from_numpy(batch_x),        
            torch.from_numpy(batch_y),        
            torch.from_numpy(local_scaler),   
        )'''


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
                      stride, normalization):
    data             = np.load(npz_path)
    positions        = data["positions"]
    train_test_split = int(data["train_test_split"])

    train_end = train_test_split - prediction_length
    val_start = train_end - context_length

    mean = float(positions[:, :train_end].mean())
    std  = float(positions[:, :train_end].std())

    print(f"normalization : {normalization}")
    print(f"mean          : {mean:.4f}  std: {std:.4f}")
    if normalization == "local":
        print("  (local: per-sample scaler = mean(|context|))")

    train_ctx, train_fcst = _make_windows(
        positions, context_length, prediction_length,
        stride=stride, T=train_end,
    )

    val_ctx  = positions[:val_size, val_start:val_start + context_length]
    val_fcst = positions[:val_size, val_start + context_length:train_test_split]

    test_ctx  = positions[:test_size, train_end:train_end + context_length]
    test_fcst = positions[:test_size, train_end + context_length:train_test_split + prediction_length]

    train_dataset = MDCCDMDataset(train_ctx,  train_fcst, mean, std, normalization)
    val_dataset   = MDCCDMDataset(val_ctx,    val_fcst,   mean, std, normalization)
    test_dataset  = MDCCDMDataset(test_ctx,   test_fcst,  mean, std, normalization)

    print(f"train samples : {len(train_dataset)}")
    print(f"val   samples : {len(val_dataset)}")
    print(f"test  samples : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=1)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=1)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=1)

    return train_loader, val_loader, test_loader, mean, std