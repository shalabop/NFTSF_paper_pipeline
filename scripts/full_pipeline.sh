python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/double_well_test.npz --out results/arima/double_well_25_25.npz --n_jobs 8
#!/usr/bin/env bash
#SBATCH --job-name=fullpipeline
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

source /packages/apps/mamba/2.0.8/etc/profile.d/conda.sh
conda activate /home/.conda/envs/venv310
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

echo "========= ARIMA ========="

python eval/ARIMA/run_auto_arima.py -c configs/arima/25_25.yaml --input data/double_well_test.npz --out results/arima/double_well_25_25.npz --n_jobs 8

echo "========= CSDI ========="

python train/CSDI/train_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  data/alanine_phi_train.npz \
    --out    checkpoints_25_25_dummy/csdi/alanine_phi \
    --device cuda:0

python eval/CSDI/forecast_csdi.py \
    --config configs/csdi_train/alanine_phi_25_25.yaml \
    --input  DATA/alanine_phi_test.npz \
    --ckpt   checkpoints_25_25_dummy/csdi_with_early_stopping/alanine_phi/model.pth \
    --out    results/csdi_with_early_stopping/alanine_phi_25_25.npz \
    --device cuda:0

echo "========= NFTSF ========="

python NFTSF/train_model.py \
    --data_path alanine_phi_train.npy \
    --data_format multi_sim \
    --n_past 25 --n_future 25 \
    --flow_blocks 6 --hidden_units 64 --hidden_layers 1,2 \
    --tail_bound 30 \
    --epochs 1000 --learning_rate 1e-3 \
    --batch_size 4096 --stride 5 \
    --weight_decay 1e-5 --seed 42 \
    --normalize --use_scheduler \
    --output_dir results/alanine_phi \
    --model_name nftsf_alanine_phi

python eval/NF/forecast_nf.py \
    --model_path   checkpoints_25_25/nf/alanine_phi_25_25.pth     \
    --config configs/nf/alanine_phi_25_25_forecast.yaml \
    --data_path    data/alanine_phi_test.npz    \
    --out          results/nf/alanine_phi_25_25.npz \
    --context_length     25    \
    --prediction_length     25   \
    --n_samples    1000 \
    --train_test_split 900 \
    --test_size 3000

echo "========= TSDIff ========="
python train/TSDiff/train_tsdiff.py \
        --dataset_path gluonts_datasets/alanine_phi \
        --config configs/tsdiff_train/alanine_phi_25_25.yaml \
        --out_dir checkpoints_25_25_dummy/tsdiff/alanine_phi

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/alanine_phi_mse_03_25_25.yaml  \
        --dataset_path gluonts_datasets/alanine_phi \
        --out results/tsdiff_mse/alanine_phi_mse_03_25_25.npz

python eval/TSDiff/forecast_tsdiff.py   \
        --config configs/tsdiff_forecast/alanine_phi_q_4_50_50.yaml  \
        --dataset_path gluonts_datasets/alanine_phi \
        --out results/tsdiff_q/alanine_phi_q_4_50_50.npz

echo "========= TSDIff Cond ========="

python train/TSDiff/train_cond_tsdiff.py \
        --dataset_path gluonts_datasets/alanine_phi \
        --config configs/tsdiff_cond_train/alanine_phi_25_25.yaml \
        --out_dir checkpoints_25_25_dummy/tsdiff_cond/alanine_phi
    
python eval/TSDiff/forecast_tsdiff_cond.py   \
        --config configs/tsdiff_forecast/alanine_phi_cond_25_25.yaml  \
        --dataset_path gluonts_datasets/alanine_phi \
        --out results/tsdiff_cond/alanine_phi_25_25.npz