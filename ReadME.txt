Install all the required Python dependencies:
pip install -r requirements.txt
To run inference on a test video file, head into the directory the command: 

python run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/example_03.mp4