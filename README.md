
This project is fork https://github.com/ultralytics/yolov5

Install
```
  $ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip install git+https://github.com/JunnYu/mish-cuda
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch
  $ pip install -r requirements.txt
```

Check CUDA
```
  $ python -c "import torch;print(torch.cuda.is_available())"
```