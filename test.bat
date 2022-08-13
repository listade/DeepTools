
call "env\Scripts\activate.bat"

echo Running detection..
python -m detect.inference --input "input" --weights "weights\car.pt" --cfg "cfg\yolov4-p5.yaml" --save-img --augment 

echo Running segmentation..
python -m semantic.inference --input "input" --weights "weights\lake.ckpt" 
