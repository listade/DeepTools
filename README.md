
The project is fork https://github.com/ultralytics/yolov5

Install
```
  $ pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
  $ pip install git+https://github.com/JunnYu/mish-cuda
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch
  $ pip install -r requirements.txt

  $ python -c "import torch;print(torch.cuda.is_available())"
```
Usage
```
  # Train
  $ python -m semantic.train --train ./train --valid ./valid --epochs 300 --name my_weights

  # Inference
  $ python -m semantic.segment --input ./images --weights ./my_weights.ckpt

  # Tilling images
  $ python -m detect.crop --images ./images --labels ./labels --img-size 320

  # Train from scratch
  $ python -m detect.train --data ./data/data.yaml --cfg ./cfg/yolov4-p5.yaml --hyp ./data/hyp.scratch.yaml --epochs 300

  # Train for tune
  $ python -m detect.train --data ./data/data.yaml --cfg ./cfg/yolov4-p5.yaml --hyp ./data/hyp.finetune.yaml --weights ./my_weights.pt --epochs 100 

  # Inference
  $ python -m detect.detect --input ./images --output ./labels --weights ./my_weights.pt --save-img

```