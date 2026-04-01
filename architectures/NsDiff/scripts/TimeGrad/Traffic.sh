SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR}"
source "$REPO_ROOT/runner.sh"
export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/TimeGrad.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Traffic " \
   --residual_layers=4 \
   --device="cuda:0" \
   --batch_size=16 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'