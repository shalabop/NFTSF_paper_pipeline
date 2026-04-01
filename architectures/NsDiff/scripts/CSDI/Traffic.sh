PYTHON_RUNNER=(conda run -n unified_tsf python)
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/CSDI.py \
   config_wandb --project=3108Diffusion \
   --dataset_type="Traffic" \
   --device="cuda:1" \
   --batch_size=4 \
   --horizon=1 \
   --layers=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[3]'
