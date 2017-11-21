# Object Detection CNN
Object detection using convolutional neural networks with already trained YOLO (You Only Look Once) model.

Input image:
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33058832-7ef756f4-cecc-11e7-82ac-0dab5f8b0cb3.jpg" width="400"></p>

Output image:
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33058875-b9613cc4-cecc-11e7-8153-5212e23db434.jpg" width="400"></p>


## Notes:

### Trained model:
<ul>
<li>Download trained model (size >25 MB) from <a href="https://pjreddie.com/media/files/yolo-voc.weights">here</a>.</li>
<li>Change name to "yolo.h5"</li>
<li>Locate in /model_data/yolo.h5</li>
</ul>

### Input image
For each image:
<ul>
<li>Update shape in car_detection_yolo.py line 96</li>
<li>Update name in car_detection_yolo.py line 126</li>
</ul>
