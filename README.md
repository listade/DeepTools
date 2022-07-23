  ```
  $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116` 
  $ pip install git+https://github.com/JunnYu/mish-cuda

  $ python -m detect.inference -h

  usage: inference.py [-h] [--input <path-to-images>] [--output <path-to-txt>]
                           [--weights <path-to-*.pt>] [--device <cuda|cpu>]
                           [--conf-thres <0-1.0>] [--iou-thres <0-1.0>]
                           [--img-size <px>] [--overlap <px>]

      optional arguments:
        -h, --help            show this help message and exit
        --input <path-to-images>
        --output <path-to-txt>
        --weights <path-to-*.pt>
        --device <cuda|cpu>
        --conf-thres <0-1.0>
        --iou-thres <0-1.0>
        --img-size <px>
        --overlap <px>
```