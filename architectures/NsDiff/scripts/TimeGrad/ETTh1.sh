SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
source "$REPO_ROOT/runner.sh"
export PYTHONPATH=./:/notebooks/pytorchtimseries
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/TimeGrad.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="ETTh2 " \
   --device="cuda:4" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'