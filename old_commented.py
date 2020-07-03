# Import packages
import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
from random import randint

# VideoStream class to handle streaming of video from webcam in separate processing thread
# Source - Adrian Rosebrock, PyImageSearch: https://www.pyimagesearch.com/2015/12/28/increasing-raspberry-pi-fps-with-python-and-opencv/
class VideoStream:
    """Camera object that controls video streaming from the Picamera"""
    def __init__(self, resolution=(640,480), framerate=30):
        # Initialize the PiCamera and the camera image stream
        self.stream = cv2.VideoCapture(0)
        #ret = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        ret = self.stream.set(3,resolution[0])
        ret = self.stream.set(4,resolution[1])
            
        # Read first frame from the stream
        (self.grabbed, self.frame) = self.stream.read()

    # Variable to control when the camera is stopped
        self.stopped = False

    def start(self):
    # Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Close camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
    # Return the most recent frame
        return self.frame

    def stop(self):
    # Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
modeldir="starter"
graph='detect.tflite'
labels='labelmap.txt'
threshold=0.5
resolution='640x480'

MODEL_NAME = modeldir
GRAPH_NAME = graph
LABELMAP_NAME = labels
min_conf_threshold = float(threshold)
resW, resH = resolution.split('x')
imW, imH = int(resW), int(resH)

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.set_num_threads(3)
interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW, imH), framerate=30).start()
time.sleep(1)


# Trackers
trackers = []
failed = []


# Tracker things
multiTracker = cv2.MultiTracker_create()
run_detector = True
bboxes = [] # Only update on detector
colors = []
predictions = [] # (id, object, percentage)
last_id = 0


ids_checked = []
old_checks = []


frame_counter = 0
fail_counter = 0

state_change = True

# cv2.setNumThreads(2)

"""


Guardar objetos durante x tempo e recuperalos


Correr detetor antes de deteção e comparar centroid (já a fazer)
Correr detetor depois de deteção e comparar centroid (Mais eficiente devido a queda de fps do detetor)



"""

def checkForOlds(box, object_name):
    # Return an old id or clear is pass the time
    
    for obj in old_checks:
        # 0 box 1 color 2 prediction
        
        # Check time ???
        
        
        # WHAT TIME IMPLEMENTS ???
        
        
        if obj[2][1] == object_name:
            
            box1 = obj[0]
            
            box_centroid = (int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2)))
            box1_centroid = (int(box1[0]+(box1[2]/2)), int(box1[1]+(box1[3]/2)))
    
            if box_centroid[0] - box1_centroid[0] < 20 or box_centroid[0] - box1_centroid[0] > -20:
                if box_centroid[1] - box1_centroid[1] < 20 or box_centroid[1] - box1_centroid[1] > -20:
                    # Close
                    # Is the same =O
                    
                    
                    # Check time ???
                    
                    
                    return obj
    
    print("Lost a " + object_name)
    
    return None
    


# Check if a object alreay exists and return the color and old prediction
def getData(box, object_name):
    
    for i, box1 in enumerate(bboxes):
        
        if object_name == predictions[i][1]:
        
            box_centroid = (int(box[0]+(box[2]/2)), int(box[1]+(box[3]/2)))
            box1_centroid = (int(box1[0]+(box1[2]/2)), int(box1[1]+(box1[3]/2)))
            
            if box_centroid[0] - box1_centroid[0] < 20 or box_centroid[0] - box1_centroid[0] > -20:
                if box_centroid[1] - box1_centroid[1] < 20 or box_centroid[1] - box1_centroid[1] > -20:
                    # Close
                    # Is the same =O
                    
                    object_id = predictions[i][0]
                    ids_checked.append(object_id)
                    
                    # Return id of the object ? or i in list <-
                    return i
            
    return -1
   
def createTraker(frame):
    for bbox in bboxes:
        multiTracker.add(cv2.TrackerMedianFlow_create(), frame, bbox) # MOSSE Tracker

#for frame1 in camera.capture_continuous(rawCapture, format="bgr",use_video_port=True):
while True:
    
    #print(str(imW) + "x" + str(imH))
    #for i, predict in enumerate(predictions):
    #    print(str(predict[0]) + ': ' + str(bboxes[i]))
    
    frame_counter += 1
    if (frame_counter > 50):
        run_detector = True
        state_change = True
        frame_counter = 0

    # Start timer (for calculating frame rate)
    t1 = cv2.getTickCount()

    # Grab frame from video stream
    frame = videostream.read()

    # Acquire frame and resize to expected shape [1xHxWx3]
    #frame = frame1.copy()
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_resized = cv2.resize(frame_rgb, (width, height))
    #input_data = np.expand_dims(frame_resized, axis=0)

    # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
    #if floating_model:
     #   input_data = (np.float32(input_data) - input_mean) / input_std

    if run_detector:
    
        if state_change:
            print ("Detecting")
            state_change = False
        
        new_bboxes = []
        new_colors = []
        new_predictions = []
       # new_trackers = []
       # new_faileds = []
        
        checked = []
        
        #frame = frame1.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
        

        # Perform the actual detection by running the model with the image as input
        interpreter.set_tensor(input_details[0]['index'],input_data)
        interpreter.invoke()

        #print(output_details)

        # Retrieve detection results
        boxes = interpreter.get_tensor(output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = interpreter.get_tensor(output_details[1]['index'])[0] # Class index of detected objects
        scores = interpreter.get_tensor(output_details[2]['index'])[0] # Confidence of detected objects
        #num = interpreter.get_tensor(output_details[3]['index'])[0]  # Total number of detected objects (inaccurate and not needed)
        
        #print(classes)

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                
                ##### Check if already exists
                original_i = getData((xmin, ymin, xmax-xmin, ymax-ymin), object_name)
                
                new_box = (xmin, ymin, xmax-xmin, ymax-ymin)
                new_color = (randint(64, 255), randint(64, 255), randint(64, 255))
                
                new_predict = (last_id, object_name, int(scores[i]*100)) ## Do nothing
                
                '''

                Corre detector -> lista de objetos
                    Se um objeto tiver uma box de centroid próximo -> é o mesmo
                        faz nada
                    se não -> Inicia tracker

                '''
                
                
                if original_i != -1:
                        
                    if original_i not in checked:
                        
                        new_color = colors[original_i]
                        new_predict = (predictions[original_i][0], predictions[original_i][1], int(scores[i]*100)) # Update percentage
                        
                        #tracker = trackers[original_i]
                        
                        #if failed[original_i]:
                        #    tracker = cv2.TrackerMOSSE_create()
                        #    ok = tracker.init(frame, new_box)
                        
                        #tracker = cv2.TrackerMOSSE_create()
                        #ok = tracker.init(frame, new_box)
                        
                        checked.append(original_i)
                    
                    else:
                        # repeated
                        continue
                        
                        
                else:
                    # Second try
                    #second_try = checkForOlds((xmin, ymin, xmax-xmin, ymax-ymin), object_name)
                    #if original_i == -1:
                    
                    #tracker = cv2.TrackerMOSSE_create()
                    
                    #ok = tracker.init(frame, new_box)
                    
                    last_id += 1
                    #else:
                        
                    new_predict = (last_id, object_name, int(scores[i]*100))
                    
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), new_color, 2)

                # Draw label
                label = '[%d] %s: %d%%' % (i, object_name, int(scores[i]*100)) # Example: '[0] person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text
                
                # Doing tracker things
                new_bboxes.append(new_box)
                new_colors.append(new_color)
                new_predictions.append(new_predict)
                #new_trackers.append(tracker)
                #new_faileds.append(False)
        
        if (len(new_predictions) > 0): 
            run_detector = False
            state_change = True
            
            # Save old checks
            for i, predict in enumerate(predictions):
                if predict[0] not in ids_checked:
                    old_checks.append((bboxes[i], colors[i], predictions[i]))
                    
            ids_checked = []
            
            # Reset
            bboxes = new_bboxes
            colors = new_colors
            predictions = new_predictions
            #trackers = new_trackers
            #failed = new_faileds
            
            # Start tracker
            multiTracker = cv2.MultiTracker_create()
            createTraker(frame)
        
    else:
        if state_change:
            print ("Traking")
            state_change = False
        
        # Get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        
        # Run detector after x fails in sequence
        #if not success:
        #    fail_counter += 1
        #    if fail_counter > 3:
        #        run_detector = True
        #        state_change = True
        #        fail_counter = 0
        #else:
        #    fail_counter = 0
        '''
        for i, tracker in enumerate(trackers):
            ok, newbox = tracker.update(frame)
            
            prediction = predictions[i]

            if not ok:
                print('Tracker failed for object ' + prediction[1] + ' ' + str(prediction[0]))
                
                # Remove from list
                # Save for detector ?
                failed[i] = True
         '''   
            
        
        # Draw tracked objects
        for i, newbox in enumerate(boxes):
            
            xmin = int(newbox[0])
            ymin = int(newbox[1])
            xmax = int(newbox[0] + newbox[2])
            ymax = int(newbox[1] + newbox[3])
            # Need to check if overrides screen ?? I think no in this way
            
            #print (i)
            #print (colors[i])
            #print ((xmin,ymin))
            #print ((xmax,ymax))
        
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), colors[i], 2)
               
            prediction = predictions[i]
            
            label = '[%d] %s: %d%%' % (prediction[0], prediction[1], prediction[2]) # Example: '[0] person: 72%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.7, 2) # Get font size
            label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text
            
            cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0]+10, label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
            cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2) # Draw label text

            # Save last position of the boxes
            bboxes[i] = newbox
            
    # Draw framerate in corner of frame
    cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_DUPLEX,1,(255,255,0),2,cv2.LINE_AA)

    # All the results have been drawn on the frame, so it's time to display it.
    cv2.imshow('Object detector', frame)

    # Calculate framerate
    t2 = cv2.getTickCount()
    time1 = (t2-t1)/freq
    frame_rate_calc= 1/time1

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()
videostream.stop()
