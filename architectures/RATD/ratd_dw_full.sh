#!/usr/bin/env bash
#SBATCH --job-name=ratd_dw_full
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=logs/ratd_dw_full.%j.out
#SBATCH --error=logs/ratd_dw_full.%j.err


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
#python train_tcn.py --config tcn_config/double_well.yaml

# Step 2: Build reference database
#python retrieval_builder.py --config tcn_config/double_well.yaml --type encode

# Step 3: Build retrieval indices
#python retrieval_builder.py --config tcn_config/double_well.yaml --type retrieval

# Step 4: Train RATD
conda run -n unified_tsf python train_ratd.py --config ratd_double_well.yaml

# Step 5: Forecast
conda run -n unified_tsf python forecast_ratd.py --config ratd_double_well.yaml

conda run -n unified_tsf python plot_heatmap.py --results results/ratd/double_well.npz    --name   ratd  --dataset double_well    --out   ratd_double_well.pdf     --indices 1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 \
    --denorm training_data/double_well_std.npz
