# Install

Python 3.7.9

CUDA 11.6
https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_511.65_windows.exe

CUDNN for CUDA 11.x
https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.0/local_installers/11.6/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip

```
  $ pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip install git+https://github.com/JunnYu/mish-cuda
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch
  $ pip install -r requirements.txt

  $ python -c "import torch;print(torch.cuda.is_available())"
```
# Usage
  Training semantic segmentation
  ```
    $ python -m semantic.train --train ./train --valid ./valid --epochs 300 --name weights
  ```
  Semantic segmentation inference
  ```
    $ python -m semantic.inference --input ./images --weights ./weights.ckpt
  ```
  Training object detection
  ```
    $ python -m detect.train --data ./data/data.yaml --hyp ./data/hyp.scratch.yaml --epochs 300
  ```
  Object detection inference
  ```
    $ python -m detect.inference --input ./images --output ./labels --weights ./weights.pt --save-img
  ```
  Object detection result
  ![image](https://user-images.githubusercontent.com/96072580/182018468-b0f1ecc6-8221-4a7f-9bfe-6084d03b197d.png)

```

$ pylint --generated-members=cv2.*,torch.* app
------------------------------------------------------------------
Your code has been rated at 6.40/10 (previous run: 5.44/10, +0.96)
```

The project is fork https://github.com/ultralytics/yolov5
