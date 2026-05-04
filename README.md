conda create -n venv310 python=3.10
conda init
conda activate venv310
conda install pip
conda install --file requirements.txt

run NFTSF

run CSDI

run TSDiff-Cond
cd archtiectures/unconditional_time_series_diffusion
pip install -e . #python project installation

run TSDiff-Q/MSE

run ARIMA