conda create -n venv310 python=3.10
conda init
conda activate venv310
conda install pip
conda install --file requirements.txt

run NFTSF

run CSDI

run TSDiff-Cond

run TSDiff-Q/MSE

run ARIMA