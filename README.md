# Object Detection CNN
Object detection using convolutional neural networks with already trained <a href="https://pjreddie.com/darknet/yolo/">YOLO (You Only Look Once)</a> model.

<br/>
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33061502-553d2b4e-ced7-11e7-9a11-3e60e576d180.png" width="900"></p>

## Notes

### Trained model
<ul>
<li>Download trained model (size >25 MB) from <a href="https://pjreddie.com/media/files/yolo-voc.weights">here</a>.</li>
<li>Change name to "yolo.h5"</li>
<li>Deploy in /model_data folder</li>
</ul>

### Input image
For each image:
<ul>
<li>Update <b>image_shape</b> in car_detection_yolo.py line 96:</li>
  
```python
96  image_shape = (1080., 1440.) # image_shape = (Width, Height)
```

<li>Update name in car_detection_yolo.py line 126:</li>

```python
126  out_scores, out_boxes, out_classes = predict(sess, "sh_taxi.jpg") # name = "sh_taxi.jpg"
```

</ul>

<br/>

## How does it work?

### 1. CNN output
The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output. Run summary() to see whole framework architecture:
  <p align="center"><img src="https://user-images.githubusercontent.com/24521991/33063083-5dbc7662-cedc-11e7-81d7-2eac598352a8.png" width="800"></p>

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
  </ul>
  
  
```python  
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    #Filters YOLO boxes by thresholding on object and class confidence.
        
    # Step 1: Compute box scores
    box_scores = box_confidence * box_class_probs
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    filtering_mask = (box_class_scores >= threshold)
    
    # Step 4: Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)
    
    return scores, boxes, classes
```


  <ul>
  <li>Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes</li>
  </ul>
  <p align="center"><img src="https://user-images.githubusercontent.com/24521991/33065478-ed57eef8-cee3-11e7-8c96-9a67d63debc7.png" width="400"></p>

### 4. Final output
This gives you YOLO's final output:
<p align="center"><img src="https://user-images.githubusercontent.com/24521991/33065240-42b8b860-cee3-11e7-9cef-b219a932d1df.png" width="500"></p>
