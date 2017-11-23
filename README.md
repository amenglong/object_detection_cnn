# Object Detection CNN
Object detection using convolutional neural networks with already trained <a href="https://pjreddie.com/darknet/yolo/">YOLO (You Only Look Once)</a> model.

<br/>
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33061502-553d2b4e-ced7-11e7-9a11-3e60e576d180.png" width="900"></p>

## Notes

### Trained model
<ul>
<li>Download trained model (195 MB) from <a href="https://onedrive.live.com/download?cid=B667AF4A4E4BA251&resid=B667AF4A4E4BA251!41299&authkey=APOOotsS6Hskc_o">here</a>.</li>
<li>Save it in <b>/model_data</b> folder</li>
</ul>

### Input image
For every new image:
<ul>
<li>Update size:</li>
  
```python
# car_detection_yolo.py
96  image_shape = (1080., 1440.) # image_shape = (Height, Width)
```

<li>Update name:</li>

```python
# car_detection_yolo.py
126  out_scores, out_boxes, out_classes = predict(sess, "sh_taxi.jpg") # name = "sh_taxi.jpg"
```

</ul>

<br/>

## How does it work?

### 1. CNN output
The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. Run summary() to see whole framework architecture:
```python
# car_detection_yolo.py
99  yolo_model.summary()
```

### 2. Output processing
After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):    
<ul>
<li>Each cell in a 19x19 grid over the input image gives 425 numbers.</li>
<li>425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.</li>
<li>85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and and 80 is the number of classes we'd like to detect</li>
</ul>

### 3. Box selection
You then select only few boxes based on:
  <ul>
  <li>Score-thresholding: throw away boxes that have detected a class with a score less than the threshold</li> 
  
  ```python
  # car_detection_yolo.py
  def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6)
  ```

  <li>Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes</li>
  
  ```python
  # car_detection_yolo.py
  def iou(box1, box2)
  def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5)
  ```
  
  </ul>


### 4. Final output
This gives you YOLO's final output:
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33065240-42b8b860-cee3-11e7-9cef-b219a932d1df.png" width="500"></p>
