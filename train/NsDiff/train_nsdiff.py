import argparse
import json
import os
import sys
import yaml
import numpy as np
import torch
from tqdm import tqdm
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.abspath(__file__))                     
PROJECT_ROOT = os.path.abspath(os.path.join(ROOT, "..", ".."))        

sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures", "NsDiff"))  
sys.path.insert(0, os.path.join(PROJECT_ROOT, "architectures"))           
sys.path.insert(0, PROJECT_ROOT)      

from NsDiff.src.models.NsDiff import NsDiff
import NsDiff.src.layer.mu_backbone as ns_Transformer
import NsDiff.src.layer.g_backbone as G
from NsDiff.src.layer.nsdiff_utils import (
    q_sample, cal_sigma_tilde, cal_forward_noise)   
from NsDiff.src.utils.sigma import wv_sigma_trailing
from NsDiff.dataset_md import get_dataloader_md

EPS = 1e-8


def build_args(config, device):
    c = config
    label_len = c["context_length"] // 2
    return SimpleNamespace(
        seq_len = c["context_length"],
        pred_len = c["prediction_length"],
        label_len = label_len,
        device = device,
        #features = "T",
        features = None,
        enc_in = 1,
        dec_in= 1,
        #c_out= 1,
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


def process_train_batch(model, cond_pred_model, cond_pred_model_g,batch_x, batch_y, batch_x_mark, batch_y_mark,
                        pred_len, label_len, rolling_length, device):
    """
    batch_x : (B, ctx_len, 1)
    batch_y : (B, pred_len, 1) - forecast only
    """
    n = batch_x.size(0)

    y_sigma = wv_sigma_trailing(torch.cat([batch_x, batch_y], dim=1), rolling_length)
    y_sigma = y_sigma[:, -pred_len:, :] + EPS
    
    # In process_train_batch, after computing y_sigma and gx:
    

    dec_inp = torch.cat([batch_x[:, -label_len:, :],torch.zeros([n, pred_len, 1], device=device)], dim=1)
    batch_y_mark_input = torch.cat([batch_x_mark[:, -label_len:, :], batch_y_mark], dim=1)

    t = torch.randint(0, model.num_timesteps, (n // 2 + 1,), device=device)
    t = torch.cat([t, model.num_timesteps - 1 - t], dim=0)[:n]

    #y_0_hat_batch, _ = cond_pred_model(batch_x, batch_x_mark, dec_inp, batch_y_mark_input)
    ##wortht looking into the masing args
    y_0_hat_batch, _ = cond_pred_model(batch_x, None, dec_inp, None)


    gx = torch.clamp(cond_pred_model_g(batch_x),min=EPS)   
    loss_mean = (y_0_hat_batch - batch_y).square().mean()
    
    #
    loss_var  = (torch.sqrt(gx) - torch.sqrt(y_sigma)).square().mean() #loss of std
    #loss_var = (gx - y_sigma).square().mean() ##loss of var

    y_T_mean = y_0_hat_batch
    e = torch.randn_like(batch_y)

    forward_noise = cal_forward_noise(model.betas_tilde, model.betas_bar, gx, y_sigma, t)
    noise = e * torch.sqrt(forward_noise)

    sigma_tilde = cal_sigma_tilde(
        model.alphas, model.alphas_cumprod,
        model.alphas_cumprod_sum, model.alphas_cumprod_prev,
        model.alphas_cumprod_sum_prev,
        model.betas_tilde_m_1, model.betas_bar_m_1,
        gx, y_sigma, t)

    y_t_batch = q_sample(batch_y, y_T_mean,model.alphas_bar_sqrt, model.one_minus_alphas_bar_sqrt,t, noise=noise)
    
    '''print(f"y_0_hat mean: {y_0_hat_batch.mean().item():.4f}")
    print(f"y_0_hat std:  {y_0_hat_batch.std().item():.4f}")
    print(f"batch_y mean: {batch_y.mean().item():.4f}")
    print(f"batch_y std:  {batch_y.std().item():.4f}")
    print(f"gx mean: {gx.mean().item():.6f}")
    print(f"gx min:  {gx.min().item():.6f}")
    print(f"gx max:  {gx.max().item():.6f}")
    exit()'''

    #output, sigma_theta = model(batch_x, batch_x_mark, y_t_batch, y_0_hat_batch, gx, t)
    output, sigma_theta = model(batch_x, None, y_t_batch, y_0_hat_batch, gx, t)
    
    sigma_theta = sigma_theta + EPS

    kl_loss = ((e - output)).square().mean() + (sigma_tilde / sigma_theta).mean() - torch.log(sigma_tilde / sigma_theta).mean()

    #sigma_ratio = sigma_tilde / (sigma_theta + 1e-8)
    #kl_loss = 0.5 * (((e - output) ** 2 / (sigma_theta + 1e-8)).mean() +(sigma_ratio - 1 - torch.log(sigma_ratio + 1e-8)).mean())
    print(f"kl_loss: {kl_loss.item():.4f}, loss_mean: {loss_mean.item():.4f}, loss_var: {loss_var.item():.4f}")
    
    return kl_loss + loss_mean + loss_var   
    #return loss_mean
    # Weight the losses to prevent KL domination
    #lambda_kl = 1
    #lambda_mean = 1.0
    #lambda_var = 1.0
    
    #total_loss = lambda_kl * kl_loss + lambda_mean * loss_mean + lambda_var * loss_var
    #return total_loss

def evaluate(model, cond_pred_model, cond_pred_model_g,
             val_loader, pred_len, label_len, rolling_length, device):
    model.eval()
    cond_pred_model.eval()
    cond_pred_model_g.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            batch_x, batch_y, x_mark, y_mark, _ = batch
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            x_mark  = x_mark.to(device).float()
            y_mark  = y_mark.to(device).float()
            loss = process_train_batch(model, cond_pred_model, cond_pred_model_g,batch_x, batch_y, x_mark, y_mark,
                pred_len, label_len, rolling_length, device)
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    cond_pred_model.train()
    cond_pred_model_g.train()
    return total_loss / max(n_batches, 1)


def main():
    parser = argparse.ArgumentParser(description="Train NsDiff on MD trajectories")
    parser.add_argument("--config","-c", required=True)
    parser.add_argument("--input", "-i", required=True)
    parser.add_argument("--out","-o", required=True)
    parser.add_argument("--device", "-d", default="cuda:0")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    device= torch.device(args.device)
    ctx_len= int(config["context_length"])
    pred_len= int(config["prediction_length"])
    batch_size= int(config["batch_size"])
    epochs= int(config["epochs"])
    lr= config["lr"]
    patience= int(config["patience"])
    stride= int(config["stride"])
    rolling_length = int(config["rolling_length"])
    normalization= config["normalization"]
    val_interval= int(config["valid_epoch_interval"])
    val_size= int(config["val_size"])
    test_size= int(config["test_size"])
    label_len= ctx_len // 2

    print(json.dumps(config, indent=4))

    train_loader, val_loader, _ = get_dataloader_md(
        npz_path = args.input,
        context_length = ctx_len,
        prediction_length = pred_len,
        batch_size = batch_size,
        stride = stride,
        #normalization = normalization,
        val_size= val_size,
        test_size= test_size,
    )

    sample = next(iter(train_loader))
    bx, by, xm, ym, ls = sample
    print(f"batch_x shape : {bx.shape}")   
    print(f"batch_y shape : {by.shape}")   

    #np.save(os.path.join(args.out, "mean.npy"), np.array([mean]))
    #np.save(os.path.join(args.out, "std.npy"), np.array([std]))
    #np.save(os.path.join(args.out, "normalization.npy"), np.array([normalization]))
    np.save(os.path.join(args.out, "context_length.npy"), np.array([ctx_len]))
    np.save(os.path.join(args.out, "prediction_length.npy"), np.array([pred_len]))
    np.save(os.path.join(args.out, "rolling_length.npy"), np.array([rolling_length]))

    model_args = build_args(config, device)
    model = NsDiff(model_args, device).to(device)
    cond_pred_model = ns_Transformer.Model(model_args).float().to(device)
    #cond_pred_model_g = G.SigmaEstimation(ctx_len, pred_len, 1, 512).float().to(device)
    cond_pred_model_g = G.SigmaEstimation(ctx_len, pred_len,1, kernel_size=1, hidden_size=32).float().to(device)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in cond_pred_model.parameters()) +
                sum(p.numel() for p in cond_pred_model_g.parameters()))
    print(f"Total parameters : {n_params:,}")

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': cond_pred_model.parameters()},
        {'params': cond_pred_model_g.parameters()},
    ], lr=lr)
    
    '''optimizer = torch.optim.Adam([
        {'params': model.parameters(), 'lr': 0.01},
        {'params': cond_pred_model.parameters(), 'lr': 5e-4},
        {'params': cond_pred_model_g.parameters(), 'lr': 5e-5},
    ])'''

    best_val_loss  = float('inf')
    patience_count = 0
    
    train_losses = []
    val_losses=[]
    val_epochs=[]
    last_val_loss=None

    for epoch in range(1, epochs + 1):
        model.train()
        cond_pred_model.train()
        cond_pred_model_g.train()

        epoch_losses = [] 
        
        with tqdm(total=len(train_loader.dataset),desc=f"Epoch {epoch}/{epochs}") as pbar:
            for batch in train_loader:
                batch_x, batch_y, x_mark, y_mark, _ = batch
                batch_x = batch_x.to(device).float()
                batch_y = batch_y.to(device).float()
                x_mark  = x_mark.to(device).float()
                y_mark  = y_mark.to(device).float()

                optimizer.zero_grad()
                loss = process_train_batch(
                    model, cond_pred_model, cond_pred_model_g,
                    batch_x, batch_y, x_mark, y_mark,
                    pred_len, label_len, rolling_length, device)
                loss.backward()
                optimizer.step()

                pbar.update(batch_x.size(0))
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                epoch_losses.append(loss.item()) 

        avg_train = np.mean(epoch_losses)
        train_losses.append(avg_train) 

        if epoch % val_interval == 0 or epoch == 1:
            val_loss = evaluate(
                model, cond_pred_model, cond_pred_model_g,
                val_loader, pred_len, label_len, rolling_length, device)
            print(f"Epoch {epoch:4d}  "
                  f"train={avg_train:.5f}  val={val_loss:.5f}")
            
            ##saving loss
            val_losses.append(val_loss)  
            val_epochs.append(epoch)   
            last_val_loss = val_loss    

            if val_loss < best_val_loss:
                best_val_loss  = val_loss
                patience_count = 0
                torch.save(model.state_dict(),os.path.join(args.out, "model.pth"))
                torch.save(cond_pred_model.state_dict(),os.path.join(args.out, "cond_pred_model.pth"))
                torch.save(cond_pred_model_g.state_dict(),os.path.join(args.out, "cond_pred_model_g.pth"))
                print(f"  => saved (best val={best_val_loss:.5f})")
                
            else:
                
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch} "
                          f"(patience={patience})")
                    break
        else:
            val_losses.append(np.nan)           
            val_epochs.append(epoch)      
                
    #save losses
    # Save losses (only validation epochs with actual values)
    np.savez(
        f"{config['train_curv']}/loss.npz",
        train_total=np.array(train_losses),
        val_total=np.array(val_losses),      # Only validation epochs
        val_epochs=np.array(val_epochs),     # Corresponding epoch numbers
        epochs=np.arange(1, len(train_losses) + 1),
    )
    
    epochs_list=np.arange(1, len(train_losses) + 1)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_list, train_losses, 'b-', label='Train Loss')

    # Validation loss only at specific epochs (val_epochs)
    if len(val_epochs) > 0:
        plt.plot(val_epochs, val_losses, 'ro-', label='Val Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curves')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{config['train_curv']}/loss_curves.png", dpi=150)

if __name__ == "__main__":
    main()