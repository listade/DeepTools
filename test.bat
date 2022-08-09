
call "env\Scripts\activate.bat"

echo Running detection..
python -m detect.inference --input "input" --weights "weights\car.pt" --save-img --augment

echo Running segmentation..
python -m semantic.inference --input "input" --weights "weights\lake.ckpt" 
