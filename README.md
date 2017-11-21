# Object Detection CNN
Object detection using convolutional neural networks with already trained <a href="https://pjreddie.com/darknet/yolo/">YOLO (You Only Look Once)</a> model.

<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33061502-553d2b4e-ced7-11e7-9a11-3e60e576d180.png" width="900"></p>

### How does it work?
<ul>
<li>The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.</li>
<li>After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):</li>
<ul>
<li>Each cell in a 19x19 grid over the input image gives 425 numbers.</li>
<li>425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.</li>
<li>85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and and 80 is the number of classes we'd like to detect</li>
</ul>
<li>You then select only few boxes based on:</li>
<ul>
<li>Score-thresholding: throw away boxes that have detected a class with a score less than the threshold</li>
<li>Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes</li>
</ul>
<li>This gives you YOLO's final output.</li>
</ul>

## Notes

### Trained model
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
