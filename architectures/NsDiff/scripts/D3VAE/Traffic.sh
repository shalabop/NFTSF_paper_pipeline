SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
source "$REPO_ROOT/runner.sh"
export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/D3VAE.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Traffic" \
   --device="cuda:5" \
   --num_preprocess_cells=1 \
   --num_channels_enc=16 \
   --num_channels_dec=16 \
   --num_postprocess_cells=1 \
   --batch_size=1 \
   --hidden_size=16 \
   --num_layers=1 \
   --groups_per_scale=1 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'