call RefreshEnv.cmd

python -m venv "%~1\env"
call "%~1\env\Scripts\activate.bat"

pip install pip\torch-1.12.1+cu116-cp37-cp37m-win_amd64.whl
pip install pip\torchvision-0.13.1+cu116-cp37-cp37m-win_amd64.whl 
pip install pip\torchaudio-0.12.1+cu116-cp37-cp37m-win_amd64.whl
pip install lib\segmentation-models-pytorch
pip install lib\mish-cuda
pip install --no-index --find-links pip\ -r requirements.txt
pip install app\

timeout 3