#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel)"
source "$REPO_ROOT/runner.sh"
#SBATCH --job-name=ratd_sw_full
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/ratd_sw_full.%j.out
#SBATCH --error=logs/ratd_sw_full.%j.err


source /home/meahmed/venv310/bin/activate

module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false

#Verify the path before running (helps with debugging)
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"

export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

export KEOPS_CACHE_DIR=$HOME/.cache/keops
mkdir -p $KEOPS_CACHE_DIR


ls $CUDA_PATH/include/cuda.h || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "nvrtc.h not found"

ls $CUDA_PATH/lib64/libnvrtc.so* || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"


# Step 1: Train TCN
"${PYTHON_RUNNER[@]}" train_tcn.py --config tcn_config/single_well.yaml

# Step 2: Build reference database
"${PYTHON_RUNNER[@]}" retrieval_builder.py --config tcn_config/single_well.yaml --type encode

# Step 3: Build retrieval indices
"${PYTHON_RUNNER[@]}" retrieval_builder.py --config tcn_config/single_well.yaml --type retrieval

# Step 4: Train RATD
"${PYTHON_RUNNER[@]}" train_ratd.py --config ratd_single_well.yaml

# Step 5: Forecast
"${PYTHON_RUNNER[@]}" forecast_ratd.py --config ratd_single_well.yaml

"${PYTHON_RUNNER[@]}" plot_heatmap.py --results results/ratd/single_well.npz     --name   ratd  --dataset single_well    --out   ratd_single_well.pdf     --indices 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21
