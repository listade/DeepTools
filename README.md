  ```
  $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu116` 
  $ pip install git+https://github.com/JunnYu/mish-cuda
  $ pip install git+https://github.com/qubvel/segmentation_models.pytorch

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

  $ python -m segment.inference -h

    usage: inference.py [-h] [--input <path-to-images>] [--output <path-to-masks>]
                             [--weights <path-to-*.ckpt>] [--device <cuda|cpu>]
                             [--batch-size <int>] [--arch <str>] [--encoder <str>]
                             [--encoder-weights <str>] [--img-size <px>]
                             [--overlap <px>] [--shrink <0-1.0>] [--conf-thres <0-1.0>]

    optional arguments:
      -h, --help            show this help message and exit
      --input <path-to-images>
      --output <path-to-masks>
      --weights <path-to-*.ckpt>
      --device <cuda|cpu>
      --batch-size <int>
      --arch <str>
      --encoder <str>
      --encoder-weights <str>
      --img-size <px>
      --overlap <px>
      --shrink <0-1.0>
      --conf-thres <0-1.0>
```