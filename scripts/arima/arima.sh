#!/usr/bin/env bash
#SBATCH --job-name=arima
#SBATCH --time=00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=ALL
#SBATCH --partition=<your-partition>  # TODO: set to your cluster partition
#SBATCH --qos=<your-qos>  # TODO: set to your cluster QOS
#SBATCH --gres=gpu:<type>:1  # TODO: set to your GPU type (e.g. gpu:a30:1, gpu:a100:1)
#SBATCH --mem=32G
#SBATCH --output=logs/arima2.%j.out
#SBATCH --error=logs/arima2.%j.err

# TODO: activate your conda/mamba environment (e.g.: source /path/to/mamba/etc/profile.d/conda.sh)
# TODO: conda activate <your-env-name>  (e.g.: conda activate venv310)
which python

mkdir -p results/arima

which python

PROJECT_ROOT=$(pwd) 
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

module purge
module load cuda-12.8.1-gcc-12.1.0

export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/targets/x86_64-linux/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CUDA_PATH/targets/x86_64-linux/lib:$LIBRARY_PATH"
export CPATH="$CUDA_PATH/targets/x86_64-linux/include:$CPATH"

rm -rf ~/.cache/keops* 2>/dev/null
rm -rf ~/.cache/pykeops* 2>/dev/null
rm -rf /tmp/keops* 2>/dev/null
rm -rf /tmp/pykeops* 2>/dev/null

export PYKEOPS_BUILD_DIR="$SLURM_TMPDIR/pykeops_build"
export KEOPS_CACHE_FOLDER="$SLURM_TMPDIR/keops_cache"

if [ -z "$SLURM_TMPDIR" ]; then
    export PYKEOPS_BUILD_DIR="$HOME/.cache/pykeops_build"
    export KEOPS_CACHE_FOLDER="$HOME/.cache/keops_cache"
fi

mkdir -p "$PYKEOPS_BUILD_DIR"
mkdir -p "$KEOPS_CACHE_FOLDER"

echo "PyKeOps build dir: $PYKEOPS_BUILD_DIR"
echo "KeOps cache dir: $KEOPS_CACHE_FOLDER"

echo "Checking CUDA installation..."
ls $CUDA_PATH/include/cuda.h || echo "WARNING: cuda.h not found"
ls $CUDA_PATH/include/nvrtc.h || echo "WARNING: nvrtc.h not found"
ls $CUDA_PATH/lib64/libnvrtc.so* || ls $CUDA_PATH/targets/x86_64-linux/lib/libnvrtc.so* || echo "WARNING: libnvrtc.so not found"

python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/double_well_test.npz --out results/arima/double_well_25_25.npz --n_jobs 8
python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/single_well_test.npz --out results/arima/single_well_25_25.npz --n_jobs 8

python eval/ARIMA/run_auto_arima.py -c configs/arima/50_50.yaml --input data/double_well_test.npz --out results/arima/double_well_50_50.npz --n_jobs 8
python eval/ARIMA/run_auto_arima.py -c configs/arima/50_50.yaml --input data/single_well_test.npz --out results/arima/single_well_50_50.npz --n_jobs 8

python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/alanine_phi_test.npz --out results/arima/alanine_phi_25_25.npz --n_jobs 8
python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/alanine_psi_test.npz --out results/arima/alanine_psi_25_25.npz --n_jobs 8

python eval/ARIMA/run_auto_arima.py -c configs/arima/50_50.yaml --input data/alanine_phi_test.npz --out results/arima/alanine_phi_50_50.npz --n_jobs 8
python eval/ARIMA/run_auto_arima.py -c configs/arima/50_50.yaml --input data/alanine_psi_test.npz --out results/arima/alanine_psi_50_50.npz --n_jobs 8