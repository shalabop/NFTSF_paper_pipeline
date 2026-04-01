SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
source "$REPO_ROOT/runner.sh"
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/DiffusionTS.py \
   --dataset_type="ExchangeRate" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
