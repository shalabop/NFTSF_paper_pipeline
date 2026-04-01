# forecast_nsdiff.py
import argparse
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "architectures", "NsDiff"))
sys.path.insert(0, ROOT)

from models.NsDiff.src.models.NsDiff import NsDiff
import models.NsDiff.src.layer.mu_backbone as ns_Transformer
import models.NsDiff.src.layer.g_backbone as G
from models.NsDiff.src.layer.nsdiff_utils import p_sample_loop
from models.NsDiff.dataset_md import get_dataloader_md

EPS = 1e-8


def build_args(config, device):
    c = config
    label_len = c["context_length"] // 2
    return SimpleNamespace(
        seq_len= c["context_length"],
        pred_len= c["prediction_length"],
        label_len= label_len,
        device= device,
        features= "M",
        enc_in= 1,
        dec_in= 1,
        c_out= 1,
        d_model= c["d_model"],
        n_heads= c["n_heads"],
        e_layers= c["e_layers"],
        d_layers= c["d_layers"],
        d_ff= c["d_ff"],
        moving_avg= c["moving_avg"],
        timesteps= c["diffusion_steps"],
        factor= c.get("factor", 3),
        distil= c.get("distil", True),
        beta_schedule= c.get("beta_schedule", "linear"),
        beta_start= c["beta_start"],
        beta_end= c["beta_end"],
        #embed= "timeF",
        embed= "fixed",
        freq= "h",
        dropout= c.get("dropout", 0.05),
        activation= c.get("activation", "gelu"),
        output_attention= False,
        do_predict= True,
        k_z= c.get("k_z", 1e-2),
        k_cond= c.get("k_cond", 1),
        p_hidden_dims= [64, 64],
        p_hidden_layers= c.get("p_hidden_layers", 2),
        d_z= c.get("d_z", 8),
        CART_input_x_embed_dim = c.get("CART_input_x_embed_dim", 32),
    )


def denormalize(x, mean, std, normalization):
    if normalization == "zscore":
        return x * std + mean
    elif normalization == "std":
        return x * std
    return x


def generate_samples(model, cond_pred_model, cond_pred_model_g,
                     batch_x, batch_x_mark, label_len, pred_len,
                     n_samples, device, repeat_n=10):
    B = batch_x.size(0)

    # build decoder input
    dec_inp = torch.cat([
        batch_x[:, -label_len:, :],
        torch.zeros([B, pred_len, 1], device=device)
    ], dim=1)
    batch_y_mark = torch.zeros([B, label_len + pred_len, 4], device=device)

    with torch.no_grad():
        #y_0_hat, _ = cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        y_0_hat, _ = cond_pred_model(batch_x, None, dec_inp, None)

        #gx = cond_pred_model_g(batch_x) + EPS
        gx = cond_pred_model_g(batch_x) + EPS

    n_loops = n_samples // repeat_n      
    remainder = n_samples  % repeat_n      
    all_preds = []

    def run_one_group(rn):
        """Tile batch by rn, run p_sample_loop, return (B, rn, pred_len)."""
        y_0_hat_tile = y_0_hat.repeat(rn, 1, 1, 1).transpose(0, 1).flatten(0, 1)
        x_tile = batch_x.repeat(rn, 1, 1, 1).transpose(0, 1).flatten(0, 1)
        x_mark_tile= batch_x_mark.repeat(rn, 1, 1, 1).transpose(0, 1).flatten(0, 1)
        gx_tile = gx.repeat(rn, 1, 1, 1).transpose(0, 1).flatten(0, 1)
        y_T_mean = y_0_hat_tile

        with torch.no_grad():
            y_seq = p_sample_loop(
                model, x_tile, None,
                y_0_hat_tile, gx_tile, y_T_mean,
                model.num_timesteps,
                model.alphas,
                model.one_minus_alphas_bar_sqrt,
                model.alphas_cumprod,
                model.alphas_cumprod_sum,
                model.alphas_cumprod_prev,
                model.alphas_cumprod_sum_prev,
                model.betas_tilde,
                model.betas_bar,
                model.betas_tilde_m_1,
                model.betas_bar_m_1,
            )

        
        y_0 = y_seq[-1][:, :, 0]                  
        y_0 = y_0.reshape(B, rn, pred_len)        
        return y_0.cpu().numpy()

    for _ in range(n_loops):
        all_preds.append(run_one_group(repeat_n))  

    if remainder > 0:
        all_preds.append(run_one_group(remainder)) 

    samples = np.concatenate(all_preds, axis=1)    
    print(samples.shape)
    samples = samples.transpose(0, 2, 1)     
    
    print(samples.shape)     
    return samples


def main():
    parser = argparse.ArgumentParser(description="Forecast with NsDiff")
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input",  "-i", required=True)
    parser.add_argument("--ckpt",   "-k", required=True,
                        help="Checkpoint directory containing model.pth etc.")
    parser.add_argument("--out",    "-o", required=True)
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device= torch.device(args.device)
    ctx_len= int(config["context_length"])
    pred_len= int(config["prediction_length"])
    batch_size= int(config["batch_size"])
    stride= int(config["stride"])
    normalization= config["normalization"]
    test_size= int(config["test_size"])
    n_samples= int(config["num_of_samples"])
    val_size= int(config["val_size"])
    label_len      = ctx_len // 2

    print(f"context_length   : {ctx_len}")
    print(f"prediction_length: {pred_len}")
    print(f"n_samples        : {n_samples}")
    print(f"normalization    : {normalization}")

    _, _, test_loader, mean, std = get_dataloader_md(
        npz_path          = args.input,
        context_length    = ctx_len,
        prediction_length = pred_len,
        batch_size        = batch_size,
        test_size         = test_size,
        stride            = stride,
        normalization     = normalization,
        val_size          = val_size,
    )

    for fname, varname in [("mean.npy","mean"),("std.npy","std"),("normalization.npy", "normalization")]:
        path = os.path.join(args.ckpt, fname)
        if os.path.exists(path):
            val = np.load(path)[0]
            if varname == "mean": mean = float(val)
            elif varname == "std": std  = float(val)
            else:normalization = str(val)
            print(f"Loaded {varname}={val} from checkpoint")

    model_args = build_args(config, args.device)

    model = NsDiff(model_args, device).to(device)
    model.load_state_dict(torch.load(
        os.path.join(args.ckpt, "model.pth"),
        map_location=device, weights_only=False))

    cond_pred_model = ns_Transformer.Model(model_args).float().to(device)
    cond_pred_model.load_state_dict(torch.load(os.path.join(args.ckpt, "cond_pred_model.pth"),
        map_location=device, weights_only=False))

    #cond_pred_model_g = G.SigmaEstimation(ctx_len, pred_len, 1, 512).float().to(device)
    cond_pred_model_g = G.SigmaEstimation(ctx_len, pred_len,1, kernel_size=1, hidden_size=32).float().to(device)
    cond_pred_model_g.load_state_dict(torch.load(os.path.join(args.ckpt, "cond_pred_model_g.pth"),
        map_location=device, weights_only=False))

    model.eval()
    cond_pred_model.eval()
    cond_pred_model_g.eval()
    print("All models loaded")

    all_samples = []
    all_gt = []

    for batch in tqdm(test_loader, desc="Forecasting"):
        batch_x, batch_y, x_mark, y_mark, local_scaler = batch
        batch_x = batch_x.to(device).float()
        x_mark  = x_mark.to(device).float()

        samples = generate_samples(
            model, cond_pred_model, cond_pred_model_g,
            batch_x, x_mark, label_len, pred_len,
            n_samples, device, repeat_n=10)     

        gt = batch_y[:, :, 0].numpy() 

        if normalization == "local":
            sc = local_scaler.numpy()[:, :, None] 
            samples = samples * sc
            gt = gt * local_scaler.numpy()
        #else:
        #    samples = denormalize(samples, mean, std, normalization)
        #    gt = denormalize(gt, mean, std, normalization)

        all_samples.append(samples)
        all_gt.append(gt)

    samples_out = np.concatenate(all_samples, axis=0) 
    gt_out = np.concatenate(all_gt, axis=0) 

    data= np.load(args.input)
    positions= data["positions"]
    time= data["time"]
    train_test_split = int(data["train_test_split"])
    N = len(samples_out)

    ci90_lower = np.percentile(samples_out,  5, axis=2)
    ci90_upper = np.percentile(samples_out, 95, axis=2)
    ci50_lower = np.percentile(samples_out, 25, axis=2)
    ci50_upper = np.percentile(samples_out, 75, axis=2)

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    np.savez(
        args.out,
        samples= samples_out,
        ground_truth= gt_out,
        ci90_lower= ci90_lower,
        ci90_upper= ci90_upper,
        ci50_lower= ci50_lower,
        ci50_upper= ci50_upper,
        full_trajectories = positions[:N],
        time  = time,
        time_train  = time[:train_test_split],
        time_test  = time[train_test_split:train_test_split + pred_len],
        train_test_split  = train_test_split,
        prediction_length = pred_len,
        num_of_samples    = n_samples,
    )
    print(f"Saved: {args.out}")

    assert not np.isnan(samples_out).any(), "NaN in samples"
    assert not np.isinf(samples_out).any(), "Inf in samples"
    assert np.all(ci90_lower <= ci90_upper), "CI90 ordering violated"
    print("Sanity checks passed ✓")


if __name__ == "__main__":
    main()