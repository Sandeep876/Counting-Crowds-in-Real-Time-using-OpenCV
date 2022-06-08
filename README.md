# Counting-Crowds-in-Real-Time-using-OpenCV


Initially, generated image frames from the input video data and then performed the object detection using the Single Shot Detectors (SSD's) with the Mobile-Net architecture. 
And then filtered out the people class among the detected objects and labeled them. Object Tracking is initiated using the Centroid Algorithm and tracking is performed to the each object labeled in the image frame of the video.
Each object is tracked thoroughly through out the input video and the objects are counted.
The FPS and the elapsed time are calculated for each detection and the Count of the people is shown in the analyzing window.

Install all the required Python dependencies:
pip install -r requirements.txt
To run inference on a test video file, head into the directory the command: 

python run.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/Video_input.mp4
