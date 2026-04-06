import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MDNsDiffDataset(Dataset):
    #def __init__(self, contexts, forecasts, mean, std, normalization="none"):
    def __init__(self, contexts, forecasts):

        """
        contexts      : (N, context_length)    raw un-normalized
        forecasts     : (N, prediction_length) raw un-normalized
        mean          : scalar — global training mean
        std           : scalar — global training std
        normalization : "zscore" | "std" | "local" | "none"
        """
        '''assert normalization in ("zscore", "std", "local", "none"), \
            f"normalization must be 'zscore', 'std', 'local', or 'none', " \
            f"got '{normalization}'"'''
        assert len(contexts) == len(forecasts)

        self.context_length = contexts.shape[1]
        self.prediction_length = forecasts.shape[1]
        #self.normalization = normalization
        #self.mean = mean
        #self.std = std

        ctx  = contexts.astype(np.float32)
        fcst = forecasts.astype(np.float32)

        '''if normalization == "zscore":
            ctx  = (ctx  - mean) / std
            fcst = (fcst - mean) / std
        elif normalization == "std":
            ctx  = ctx  / std
            fcst = fcst / std'''

        self.contexts  = ctx
        self.forecasts = fcst

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, index):
        ctx  = self.contexts[index]   
        fcst = self.forecasts[index]  #

        '''if self.normalization == "local":
            scaler = float(np.abs(ctx).mean())
            scaler = max(scaler, 1e-8)
            ctx    = (ctx  / scaler).astype(np.float32)
            fcst   = (fcst / scaler).astype(np.float32)
            local_scaler = np.array([scaler], dtype=np.float32)
        else:
            local_scaler = np.array([1.0], dtype=np.float32)'''
        local_scaler = np.array([1.0], dtype=np.float32)

        batch_x = ctx[:, None] #(N, context_length, 1)
        batch_y = fcst[:, None] #(N, prediction_length, 1)


        #x_mark = np.zeros((self.context_length, 4), dtype=np.float32) 
        #y_mark = np.zeros((self.prediction_length, 4), dtype=np.float32)

        # In dataset_md.py:
        x_times = np.arange(self.context_length, dtype=np.float32)
        y_times = np.arange(self.context_length, self.context_length + self.prediction_length, dtype=np.float32)

        max_time = float(self.context_length + self.prediction_length)
        x_times_norm = x_times / max_time
        y_times_norm = y_times / max_time

        # Tile to create (T, 4) shape
        '''x_mark = np.tile(x_times_norm[:, None], (1, 4)).astype(np.float32)
        y_mark = np.tile(y_times_norm[:, None], (1, 4)).astype(np.float32)'''
        
        x_mark = np.zeros((self.context_length,  1), dtype=np.float32)
        y_mark = np.zeros((self.prediction_length, 1), dtype=np.float32)

        return (
            torch.from_numpy(batch_x),       
            torch.from_numpy(batch_y),        
            torch.from_numpy(x_mark),         
            torch.from_numpy(y_mark),         
            torch.from_numpy(local_scaler),  
        )


def _make_windows(positions, context_length, prediction_length, stride, T):
    """
    Slide windows with given stride over positions[:, :T].
    Returns contexts (N, ctx_len) and forecasts (N, pred_len).
    """
    seq_len = context_length + prediction_length
    ctx_list  = []
    fcst_list = []
    for start in range(0, T - seq_len + 1, stride):
        ctx_list.append(positions[:, start:start + context_length])
        fcst_list.append(positions[:, start + context_length:start + seq_len])
    return (np.concatenate(ctx_list,  axis=0),np.concatenate(fcst_list, axis=0))


def get_dataloader_md(npz_path, context_length, prediction_length,
                      batch_size, val_size, test_size,
                      stride):
    data = np.load(npz_path) ##wil be e.g double_well.npz
    positions_train = data["positions_train"]
    positions_test = data["positions_test"]
    train_test_split = int(data["train_test_split"])

    #train_end = train_test_split - prediction_length   
    #val_start = train_end - context_length             
    val_start=positions_train.shape[1]-(prediction_length+context_length)
    train_end=val_start
    print(val_start)
    #exit()

    #mean = float(positions[:, :train_end].mean())
    #std  = float(positions[:, :train_end].std())

    #print(f"normalization : {normalization}")
    #print(f"mean : {mean:.4f}  std: {std:.4f}")
    #if normalization == "local":
    #    print("  (local: per-sample scaler = mean(|context|))")

    train_ctx, train_fcst = _make_windows(positions_train[:,:val_start], context_length, prediction_length,stride=stride, T=train_end)
    
    val_ctx = positions_train[:val_size,val_start:val_start + context_length]
    val_fcst = positions_train[:val_size,val_start + context_length:]

    test_ctx = positions_test[:test_size,-(prediction_length+context_length):-prediction_length]
    test_fcst = positions_test[:test_size,-prediction_length:]
    print(test_ctx.shape)
    print(test_fcst.shape)

    train_dataset = MDNsDiffDataset(train_ctx, train_fcst)
    val_dataset   = MDNsDiffDataset(val_ctx,val_fcst)
    test_dataset  = MDNsDiffDataset(test_ctx,test_fcst)

    print(f"train samples : {len(train_dataset)}")
    print(f"val   samples : {len(val_dataset)}")
    print(f"test  samples : {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,  num_workers=1)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,shuffle=False, num_workers=1)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,shuffle=False, num_workers=1)

    #return train_loader, val_loader, test_loader, mean, std
    return train_loader, val_loader, test_loader