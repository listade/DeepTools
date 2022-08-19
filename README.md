The YOLO project fork (https://github.com/ultralytics/yolov5)

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
  Dataset creating
  ```
    $ python -m detect.annotate <path-to-images>/*.jpg
  ```
  Object detection
  ```
    $ python -m detect.inference --input ./images --weights ./best.pt --cfg cfg/yolov4-p5.yaml --save-img 
  ```
  Semantic segmentation
  ```
    $ python -m semantic.inference --input ./images --weights ./best.ckpt
  ```
  Train object detection
  ```
    $ python -m detect.train --data ./data/data.yaml --hyp ./data/hyp.scratch.yaml
  ```
  Train semantic segmentation
  ```
    $ python -m semantic.train --train ./train --valid ./valid --name weights
  ```

  ![image](https://user-images.githubusercontent.com/96072580/182018468-b0f1ecc6-8221-4a7f-9bfe-6084d03b197d.png)

```

$ pylint --generated-members=cv2.*,torch.* app
------------------------------------------------------------------
Your code has been rated at 7.29/10 (previous run: 7.04/10, +0.25)
```


