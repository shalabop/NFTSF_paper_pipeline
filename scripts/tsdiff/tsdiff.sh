
#!/usr/bin/env bash
#SBATCH --job-name=DWTSFdiff
#SBATCH --mail-user=meahmed@asu.edu
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=public
#SBATCH --qos=public
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=DW_TSFdiff.%j.out
#SBATCH --error=DW_TSFdiff.%j.err


source venv310/bin/activate
which python

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge
module load cuda-12.8.1-gcc-12.1.0

export NO_AI_TRACKING=false
PPexport CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"

export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

#export KEOPS_CACHE_DIR=$SLURM_TMPDIR/keops_cache
export KEOPS_CACHE_DIR=$HOME/.cache/keops
mkdir -p $KEOPS_CACHE_DIR


ls $CUDA_PATH/include/cuda.h || echo "cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "nvrtc.h not found"

ls $CUDA_PATH/lib64/libnvrtc.so* || echo "libnvrtc.so not found in lib64"
ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "libnvrtc.so not found in targets/x86_64-linux/lib"

python train/TSDiff/train_tsdiff.py \
        --dataset_path gluonts_datasets/double_well \
        --config configs/tsdiff_train/double_well.yaml \
        --out_dir checkpoints/tsdiff/double_well

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/double_well_q_4.yaml  \
        --dataset_path gluonts_datasets/double_well \
        --out results/tsdiff/double_well_q_4

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/double_well_mse_05.yaml  \
        --dataset_path gluonts_datasets/double_well \
        --out results/tsdiff/double_well_mse_05 