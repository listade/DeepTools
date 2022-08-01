


# Install

CUDA 11.6
https://developer.download.nvidia.com/compute/cuda/11.6.2/local_installers/cuda_11.6.2_511.65_windows.exe

CUDNN for CUDA 11.x
https://developer.download.nvidia.com/compute/redist/cudnn/v8.4.0/local_installers/11.6/cudnn-windows-x86_64-8.4.0.27_cuda11.6-archive.zip

```
  $ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip install git+https://github.com/JunnYu/mish-cuda
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch
  $ pip install -r requirements.txt

  $ python -c "import torch;print(torch.cuda.is_available())"
```
# Usage
  Training
  ```
    $ python -m semantic.train --train ./train --valid ./valid --epochs 300 --name weights
  ```
  Segmentation inference
  ```
    $ python -m semantic.inference --input ./images --weights ./weights.ckpt
  ```
  Images cropping
  ```
    $ python -m detect.crop --images ./images --labels ./labels
  ```
  Training from scratch
  ```
    $ python -m detect.train --data ./data/data.yaml --hyp ./data/hyp.scratch.yaml --epochs 300
  ```
  Training for tune
  ```
    $ python -m detect.train --data ./data/data.yaml --hyp ./data/hyp.finetune.yaml --weights ./weights.pt --epochs 100
  ```
  Detection inference
  ```
    $ python -m detect.inference --input ./images --output ./labels --weights ./weights.pt --save-img
  ```
  
  ![image](https://user-images.githubusercontent.com/96072580/182018468-b0f1ecc6-8221-4a7f-9bfe-6084d03b197d.png)

```
$ pylint --generated-members=cv2.*,torch.* semantic
------------------------------------------------------------------
Your code has been rated at 8.28/10 (previous run: 8.25/10, +0.02)

$ pylint --generated-members=cv2.*,torch.* detect
------------------------------------------------------------------
Your code has been rated at 6.31/10 (previous run: 6.15/10, +0.16)
```

The project is fork https://github.com/ultralytics/yolov5
