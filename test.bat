
call "%~1\env\Scripts\activate.bat"

set PYTHONPATH="%~1"

echo Running detection..
python -m app.detect.inference --input "%~1\input" --weights "%~1\weights\car.pt" --save-img --augment

echo Running segmentation..
python -m app.semantic.inference --input "%~1\input" --weights "%~1\weights\lake.ckpt" 
