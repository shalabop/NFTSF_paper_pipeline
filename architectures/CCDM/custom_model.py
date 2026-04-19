import os
import time
import json
from pathlib import Path
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.distributions import Categorical

import torch.nn as nn
from einops import rearrange
import properscoring as ps

from network  import Denoiser
from diffusion import DDPM

def cal_mse_loss(a, b):
    B = a.shape[0]
    loss_func = nn.MSELoss(reduction='none').to(a.device)
    inst_mse  = loss_func(a, b)
    batch_mse = torch.mean(inst_mse.contiguous().view(B, -1), dim=1)  # (B,)
    return batch_mse

def cal_mse_norm(preds, truth):
    median = np.quantile(preds, q=0.5, axis=0)   # (pred_len, D)
    return np.mean(np.square(median - truth))

def cal_mae_norm(preds, truth):
    median = np.quantile(preds, q=0.5, axis=0)   # (pred_len, D)
    return np.mean(np.abs(median - truth))

def cal_crps_norm(preds, truth):
    pred_len = truth.shape[0]
    D        = truth.shape[1]
    truth    = rearrange(truth, "L D -> (L D)")
    preds    = rearrange(preds, "N L D -> (L D) N")
    crps_per_point = ps.crps_ensemble(truth, preds)
    return np.sum(crps_per_point) / (pred_len * D)

def cal_crps_sum(preds, truth):
    truth_sum = np.sum(truth, axis=1)          # (pred_len,)
    preds_sum = np.sum(preds, axis=2).T        # (pred_len, n_samples)
    crps_per_step = ps.crps_ensemble(truth_sum, preds_sum)
    return np.mean(crps_per_step)


class DiffMTS:
    """
    Mirrors the original CCDM DiffMTS exactly, adapted for:
      - 3-tuple dataloader  (batch_x, batch_y0, _)   instead of 4-tuple
      - separate val / test loaders
      - model_best.pt checkpoint saving
    """

    def __init__(self, configs, train_loader, val_loader, test_loader,
                 out_dir: str = None):
        self.configs      = configs
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.device       = configs.device

        # ── hyper-params (same attribute names as original) ───────────────
        self.use_window_norm  = configs.use_window_norm
        self.n_steps          = configs.n_steps
        self.n_epochs         = configs.n_epochs
        self.data_name        = configs.data_name
        self.num_feat         = configs.num_feat
        self.cont_len         = configs.cont_len
        self.pred_len         = configs.pred_len
        self.parameterization = configs.parameterization
        self.step_dist        = configs.step_dist
        self.use_contrast     = configs.use_contrast      # "contrast" | "non-contrast"
        self.n_negatives      = configs.n_negatives
        self.contrast_weight  = configs.contrast_weight
        self.temperature      = configs.temperature
        self.init_lr          = configs.init_lr
        self.non_attn         = configs.non_attn

        self.out_dir = Path(out_dir) if out_dir else \
                       Path("checkpoints") / "ccdm" / configs.data_name

        self.task_name = (
            f"{self.data_name}"
            f"_I{self.cont_len}_O{self.pred_len}"
            f"_Contrastive{self.contrast_weight}"
            f"_Step{self.n_steps}_similarity"
        )
        if self.non_attn:
            self.task_name += "_NonAttn"
        print(f"Running task: {self.task_name}")

        self.denoiser  = Denoiser(configs).to(self.device)
        self.diffusion = DDPM(self.denoiser, configs).to(self.device)

        self.optimizer = optim.Adam(
            self.denoiser.parameters(),
            lr=self.init_lr,
            weight_decay=1e-6,
        )
        p1 = int(0.75 * self.n_epochs)
        p2 = int(0.9  * self.n_epochs)
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[p1, p2], gamma=0.1
        )

    def step_sampling(self, batch_size):
        if self.step_dist == "uniform":
            k_half0 = torch.randint(0, self.n_steps, (batch_size // 2,))
            k_half1 = self.n_steps - 1 - k_half0
            k = torch.cat([k_half0, k_half1], dim=0)
        elif self.step_dist == "non-uniform":
            kn    = torch.exp(1 - torch.linspace(0, 1, self.n_steps))
            probs = kn / torch.sum(kn)
            k     = Categorical(probs).sample(torch.Size((batch_size,)))
        else:
            raise NotImplementedError("Such step distribution is not valid!")
        return k.to(self.device)


    def negative_sampling(self, y0, mode):
        B      = y0.shape[0]
        y0_len = y0.shape[1]

        if mode == "variation":
            #patch_size = 8
            #n_patches  = y0_len // 8
            #for our case
            patch_size = 10
            n_patches  = y0_len // 10
            y0_patch   = y0.view(B, n_patches, patch_size, self.num_feat)
            neg_samples = torch.zeros(
                (B * self.n_negatives, y0_len, self.num_feat),
                device=y0.device
            )
            for i in range(self.n_negatives):
                idx_patch  = torch.randperm(y0_patch.shape[1])
                y0_shuffle = y0_patch[:, idx_patch, :, :].reshape(B, y0_len, self.num_feat)
                neg_indices = torch.arange(0, neg_samples.shape[0], self.n_negatives) + i
                neg_samples[neg_indices] = y0_shuffle

        elif mode == "scaling":
            # per-variate scale factors — identical to original (numpy)
            scale_down = np.random.uniform(0.0, 0.5, (self.n_negatives // 2, 1, self.num_feat))
            scale_up   = np.random.uniform(1.5, 2.0, (self.n_negatives // 2, 1, self.num_feat))
            scale      = torch.from_numpy(
                np.concatenate([scale_up, scale_down], axis=0)           # (n_neg, 1, D)
            )
            scale_rep   = scale.repeat(B, 1, 1).to(y0.device)           # (B*n_neg, 1, D)
            y0_rep      = torch.repeat_interleave(y0, self.n_negatives, dim=0)  # (B*n_neg, H, D)
            neg_samples = y0_rep * scale_rep

        elif mode == "jitter":
            neg_samples = torch.zeros(
                (B * self.n_negatives, y0_len, self.num_feat),
                device=y0.device
            )
            y0_rep      = torch.repeat_interleave(y0, self.n_negatives, dim=0)
            jit_noise   = torch.randn_like(neg_samples) * (0.3 ** 0.5)
            neg_samples = y0_rep + jit_noise

        elif mode == "cutout":
            mask_len    = int(0.1 * y0_len)
            end_point   = y0_len - mask_len
            start_point = torch.randint(0, end_point, (self.n_negatives,))
            neg_samples = torch.zeros(
                (B * self.n_negatives, y0_len, self.num_feat),
                device=y0.device
            )
            for i in range(self.n_negatives):
                y0_mask = y0.clone()
                y0_mask[:, start_point[i]:start_point[i] + mask_len, :] = 0.0
                neg_indices = torch.arange(0, neg_samples.shape[0], self.n_negatives) + i
                neg_samples[neg_indices] = y0_mask

        else:
            raise NotImplementedError(f"No such negative sampling mode: {mode}!")

        return neg_samples.float()


    def cal_contrastive_loss(self, neg_samples, x, k, noise, loss_type="regression"):
        B            = x.shape[0]
        n_negatives  = neg_samples.shape[0] // B

        # original passes noise=None → independent noise per negative
        if noise is None:
            neg_noise = torch.randn_like(neg_samples)
        else:
            neg_noise = torch.repeat_interleave(noise, n_negatives, dim=0)

        neg_k  = torch.repeat_interleave(k,    n_negatives, dim=0)
        neg_x  = torch.repeat_interleave(x,    n_negatives, dim=0)
        neg_yk = self.diffusion.q_sample(neg_samples, neg_k, neg_noise)
        neg_pred = self.denoiser(neg_x, neg_yk, neg_k)                  # (B*n_neg, H, D)

        if loss_type == "regression":
            neg_loss = cal_mse_loss(neg_pred, neg_noise).view(B, n_negatives)
        else:
            # "similarity" branch (what the original calls with loss_type="similarity")
            neg_pred_flat  = neg_pred.reshape(B * n_negatives, -1)
            neg_noise_flat = neg_noise.reshape(B * n_negatives, -1)
            neg_loss = (
                F.cosine_similarity(neg_pred_flat, neg_noise_flat, dim=1)
                 .unsqueeze(1)
                 .view(B, n_negatives)
            )

        return neg_loss


    def cal_train_loss(self, x, y0, x_mark, y0_mark, loss_type="regression"):
        """
        x  : (B, L, D)
        y0 : (B, H, D)
        loss_type : "regression" | "similarity"
                    original always calls with "similarity"
        """
        B     = x.shape[0]
        k     = self.step_sampling(B)
        noise = torch.randn_like(y0)

        yk     = self.diffusion.q_sample(y0, k, noise)
        pred_k = self.denoiser(x, yk, k, x_mark, y0_mark)
        #pred_k = self.denoiser(x, yk, k, None, None)

        if self.parameterization == "noise":
            target = noise
        elif self.parameterization == "y0":
            target = y0
        else:
            raise NotImplementedError(f"No such parameterization: {self.parameterization}!")

        denoise_loss = cal_mse_loss(pred_k, target)                    # (B,)

        if self.use_contrast == "non-contrast":
            return torch.mean(denoise_loss), torch.zeros(1, device=self.device)

        neg_variation_samples = self.negative_sampling(y0, mode="variation")
        neg_scale_samples     = self.negative_sampling(y0, mode="scaling")

        neg_variation_loss = self.cal_contrastive_loss(neg_variation_samples, x, k, noise=None, loss_type=loss_type)
        neg_scale_loss     = self.cal_contrastive_loss(neg_scale_samples,     x, k, noise=None, loss_type=loss_type)

        if loss_type == "regression":
            pos_loss      = denoise_loss.view(B, 1)
            contrast_loss = -torch.cat(
                [pos_loss, neg_variation_loss, neg_scale_loss], dim=1)   # (B, 1+2N)
            infonce_loss  = -torch.log(
                torch.softmax(contrast_loss / self.temperature, dim=1)[:, 0])

        else:   # "similarity"
            pred_k_flat  = pred_k.reshape(B, -1)
            target_flat  = target.reshape(B, -1)
            pos_loss     = F.cosine_similarity(pred_k_flat, target_flat, dim=1).unsqueeze(1)
            contrast_loss = torch.cat(
                [pos_loss, neg_variation_loss, neg_scale_loss], dim=1)   # (B, 1+2N)
            infonce_loss  = -torch.log(
                torch.softmax(contrast_loss / self.temperature, dim=1)[:, 0])

        return torch.mean(denoise_loss), torch.mean(infonce_loss)


    def load_pretrained_weights(self, model_path: str, refine_epochs: int):
        """Load Stage-1 checkpoint and re-init optimizer for fine-tuning."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pretrained weights not found: {model_path}")
        weight_dict = torch.load(model_path, map_location=self.device)
        self.denoiser.load_state_dict(weight_dict)
        self.denoiser.train()
        self.n_epochs     = refine_epochs
        self.optimizer    = optim.Adam(self.denoiser.parameters(), lr=2e-5)
        self.lr_scheduler = CosineAnnealingLR(self.optimizer, T_max=refine_epochs)
        print(f"Loaded pretrained weights: {model_path}")


    def train(self, is_refine: bool = False, model_path: str = None,
              refine_epochs: int = 30, loss_path: str = None):
        """
        is_refine=False : end-to-end training for self.n_epochs epochs.
        is_refine=True  : two-stage — load model_path, fine-tune refine_epochs.
        """
        weight_dir = self.out_dir
        weight_dir.mkdir(parents=True, exist_ok=True)

        if is_refine:
            if model_path is None:
                raise ValueError("model_path must be provided for two-stage training.")
            self.load_pretrained_weights(model_path, refine_epochs)
            print(f"Fine-tuning stage is beginning for {refine_epochs} epochs.")

        epoch_loss   = {"Denoising": [], "Contrastive": [], "Total": []}
        best_val_loss = float("inf")
        valid_epoch_interval = getattr(self.configs, "valid_epoch_interval", 10)
        save_epoch_interval  = getattr(self.configs, "save_epoch_interval",  50)

        print("Training stage is beginning!")
        for epoch_no in range(self.n_epochs):
            self.denoiser.train()
            batch_loss = {"Denoising": [], "Contrastive": [], "Total": []}
            start_time = time.time()

            #for batch_x, batch_y0, _ in self.train_loader:
            for batch_x, batch_y0, local_scalr, x_mark, y0_mark in self.train_loader:
                self.optimizer.zero_grad()
                batch_x  = batch_x.float().to(self.device)
                batch_y0 = batch_y0.float().to(self.device)
                batch_x_mark, batch_y0_mark = x_mark, y0_mark

                if self.use_window_norm:
                    batch_x, batch_y0, _, _ = self.instance_normalization(batch_x, batch_y0)

                '''denoise_loss, contrast_loss = self.cal_train_loss(
                    batch_x, batch_y0,
                    x_mark=None, y0_mark=None,
                    loss_type="similarity",
                )'''
                
                denoise_loss, contrast_loss = self.cal_train_loss(
                    batch_x, batch_y0,
                    x_mark=batch_x_mark, y0_mark=batch_y0_mark,
                    loss_type="similarity",
                )
                total_loss = denoise_loss + self.contrast_weight * contrast_loss
                total_loss.backward()
                self.optimizer.step()

                batch_loss["Denoising"].append(denoise_loss.item())
                batch_loss["Contrastive"].append(contrast_loss.item())
                batch_loss["Total"].append(total_loss.item())

            self.lr_scheduler.step()
            end_time = time.time()

            epoch_loss["Denoising"].append(np.mean(batch_loss["Denoising"]))
            epoch_loss["Contrastive"].append(np.mean(batch_loss["Contrastive"]))
            epoch_loss["Total"].append(np.mean(batch_loss["Total"]))

            print(
                f"[epoch {epoch_no}/{self.n_epochs}] "
                f"Total: {epoch_loss['Total'][-1]:.5f}  "
                f"Denoising: {epoch_loss['Denoising'][-1]:.5f}  "
                f"Contrastive: {epoch_loss['Contrastive'][-1]:.5f}  "
                f"time: {end_time - start_time:.1f}s"
            )

            # ── periodic epoch checkpoint (mirrors original) ──
            if (epoch_no % save_epoch_interval == 0) or (epoch_no == self.n_epochs - 1):
                torch.save(
                    self.denoiser.state_dict(),
                    weight_dir / f"epoch{epoch_no}.pt",
                )

            # ── val loss + best model (our addition) ──
            if (epoch_no % valid_epoch_interval == 0) or (epoch_no == self.n_epochs - 1):
                val_loss = self._val_loss()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        self.denoiser.state_dict(),
                        weight_dir / "model_best.pt",
                    )
                    print(f"  ↳ best model saved (val={val_loss:.5f})")

        # ── save loss curves ──
        epochs_arr = np.arange(self.n_epochs)
        save_path  = Path(loss_path) if loss_path else weight_dir / "training_loss.npz"
        np.savez(
            save_path,
            train_total     = np.array(epoch_loss["Total"]),
            train_denoise   = np.array(epoch_loss["Denoising"]),
            train_contrast  = np.array(epoch_loss["Contrastive"]),
            epochs          = epochs_arr,
        )

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(epochs_arr, epoch_loss["Total"],       label="Total")
        ax.plot(epochs_arr, epoch_loss["Denoising"],   label="Denoising")
        ax.plot(epochs_arr, epoch_loss["Contrastive"], label="Contrastive")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.legend(); ax.set_title(self.task_name)
        fig.savefig(weight_dir / "training_loss.png", bbox_inches="tight")
        plt.close(fig)

        print(f"Training complete. Weights in {weight_dir}/")


    @torch.no_grad()
    def _val_loss(self) -> float:
        """Mean denoising loss over the validation set."""
        self.denoiser.eval()
        losses = []
        for batch_x, batch_y0,scalr, x_mark, y0_mark in self.val_loader:
            batch_x  = batch_x.float().to(self.device)
            batch_y0 = batch_y0.float().to(self.device)
            x_mark, y0_mark = x_mark.float().to(self.device), y0_mark.float().to(self.device)
            if self.use_window_norm:
                batch_x, batch_y0, _, _ = self.instance_normalization(batch_x, batch_y0)
            '''loss, _ = self.cal_train_loss(
                batch_x, batch_y0, x_mark=None, y0_mark=None, loss_type="similarity"
            )'''
            
            loss, _ = self.cal_train_loss(
                batch_x, batch_y0, x_mark=x_mark, y0_mark=y0_mark, loss_type="similarity"
            )
            losses.append(loss.item())
        self.denoiser.train()
        return float(np.mean(losses))


    def load_trained_weights(self, path: str):
        """Load a checkpoint for inference (mirrors load_trained_weights)."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights not found: {path}")
        self.denoiser.load_state_dict(torch.load(path, map_location=self.device))
        self.denoiser.eval()
        print(f"Loaded trained weights: {path}")

    # ── test / evaluation ─────────────────────────────────────────────────────

    @torch.no_grad()
    def test(self, path: str, n_samples: int = 100, save_outputs: bool = False):
        """
        path        : checkpoint .pt file to load
        n_samples   : number of diffusion samples per window
        save_outputs: if True, pickle predictions for plotting
        """
        self.load_trained_weights(path)
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        all_metrics = {}
        all_outputs = {}
        test_no     = 0

        for batch_no, (batch_x, batch_y0, x_mark, y0_mark) in enumerate(self.test_loader):
            start_time = time.time()
            batch_x  = batch_x.float().to(self.device)
            batch_y0 = batch_y0.float().to(self.device)
            x_mark, y0_mark = x_mark.float().to(self.device), y0_mark.float().to(self.device)

            if self.use_window_norm:
                batch_x, _, x_mean, x_std = self.instance_normalization(batch_x, batch_y0)

            # (B*n_samples, H, D)
            '''pred_out = self.diffusion.sampling(n_samples, batch_x,
                                               x_mark=None, y0_mark=None)'''
                                               
            pred_out = self.diffusion.sampling(n_samples, batch_x,
                                               x_mark=x_mark, y0_mark=y0_mark)

            if self.use_window_norm:
                pred_out = self.instance_denormalization(pred_out, x_mean, x_std)

            B        = batch_x.shape[0]
            pred_out = pred_out.view(B, n_samples, self.pred_len, self.num_feat)

            for i in range(B):
                pred_np = pred_out[i].cpu().numpy()      # (S, H, D)
                gt_np   = batch_y0[i].cpu().numpy()      # (H, D)
                ctx_np  = batch_x[i].cpu().numpy()       # (L, D)

                mse      = cal_mse_norm(pred_np, gt_np)
                mae      = cal_mae_norm(pred_np, gt_np)
                crps     = cal_crps_norm(pred_np, gt_np)
                crps_sum = cal_crps_sum(pred_np, gt_np)
                all_metrics[str(test_no)] = {
                    "mse": float(mse), "mae": float(mae),
                    "crps": float(crps), "crps_sum": float(crps_sum),
                }

                if save_outputs:
                    all_outputs[str(test_no)] = {
                        "Predictions" : pred_np,
                        "Ground truth": gt_np,
                        "Conditions"  : ctx_np,
                    }

                print(f"  {test_no:4d}: MSE={mse:.5f}  MAE={mae:.5f}  "
                      f"CRPS={crps:.5f}  CRPS_sum={crps_sum:.5f}")
                test_no += 1

            end_time = time.time()
            print(f"Batch {batch_no}: {end_time - start_time:.1f}s")

        mse_avg      = np.mean([v["mse"]      for v in all_metrics.values()])
        mae_avg      = np.mean([v["mae"]      for v in all_metrics.values()])
        crps_avg     = np.mean([v["crps"]     for v in all_metrics.values()])
        crps_sum_avg = np.mean([v["crps_sum"] for v in all_metrics.values()])
        print(f"\n[Final] MSE={mse_avg:.5f}  MAE={mae_avg:.5f}  "
              f"CRPS={crps_avg:.5f}  CRPS_sum={crps_sum_avg:.5f}")

        with open(output_dir / f"{self.task_name}_metrics.json", "w") as f:
            json.dump(all_metrics, f, indent=4)

        if save_outputs:
            import pickle
            with open(output_dir / f"{self.task_name}_outputs.pkl", "wb") as f:
                pickle.dump(all_outputs, f)

        return all_metrics