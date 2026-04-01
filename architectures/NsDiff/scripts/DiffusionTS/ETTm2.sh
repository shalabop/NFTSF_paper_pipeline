PYTHON_RUNNER=(conda run -n unified_tsf python)
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/DiffusionTS.py \
   --dataset_type="ETTm2" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=192 \
   --windows=168 \
   runs --seeds='[1, 2, 3]'
