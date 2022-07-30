


# Install
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

The project is fork https://github.com/ultralytics/yolov5
