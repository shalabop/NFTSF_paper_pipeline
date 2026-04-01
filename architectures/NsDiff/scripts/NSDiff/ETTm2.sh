PYTHON_RUNNER=(conda run -n unified_tsf python)
export PYTHONPATH=./
CUDA_DEVICE_ORDER=PCI_BUS_ID \
"${PYTHON_RUNNER[@]}" ./src/experiments/NsDiff.py \
   --dataset_type="ETTm2" \
   --device="cuda:0" \
   --batch_size=32 \
   --horizon=1 \
   --pred_len=24 \
   --windows=168 \
   --rolling_length=24 \
   --epochs=50 \
   --patience=10 \
   runs --seeds='[1, 2, 3]'
