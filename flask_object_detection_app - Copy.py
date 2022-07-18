from flask import Flask, render_template, request, session, Response
#from flask import Flask, render_template, request, url_for, redirect
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
from os.path import join, dirname, realpath
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = os.path.join('static', 'uploads')
#UPLOAD_FOLDER = join(dirname(realpath(__file__)), 'static\\uploads\\')
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

app.secret_key = 'tornado'

# YOLO object detection function
def detect_object(uploaded_image_path):
    # Loading image
    image = cv2.imread(uploaded_image_path)
    # the neural network configuration
    config_path = "D:\Data Analytics\Self Study\YOLO Model\yolov3\yolov3.cfg"
    # the YOLO net weights file
    weights_path = "D:\Data Analytics\Self Study\YOLO Model\yolov3\yolov3.weights"
    # weights_path = "weights/yolov3-tiny.weights"

    # loading all the class labels (objects)
    labels = open("D:\Data Analytics\Self Study\YOLO Model\yolov3\coco.names").read().strip().split("\n")
    # generating colors for each object for later plotting
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # load the YOLO network
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    # This will normalize pixel values to range from 0 to 1, resize the image to (416, 416), and reshape it. 

    # Making Predictions
    # sets the blob as the input of the network
    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")

    #We need to iterate over the neural network outputs and discard any object with confidence less than the CONFIDENCE parameter we specified earlier (i.e 0.5 or 50%).
    font_scale = 1
    thickness = 1
    boxes, confidences, class_ids = [], [], []
    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Drawing Detected Objects
    # loop over the indexes we are keeping
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]
        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

    # Write output image (object detection output)
    output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'output_image.jpg')
    cv2.imwrite(output_image_path, image)
 
    return(output_image_path)

@app.route('/')
def index():
    return render_template('index_upload_and_display_image.html')
 
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
 
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
 
        return render_template('index_upload_and_display_image_page2.html')
 
@app.route('/show_image')
def displayImage():
    img_file_path = session.get('uploaded_img_file_path', None)
    return render_template('show_image.html', user_image = img_file_path)
 
@app.route('/detect_object')
def detectObject():
    uploaded_image_path = session.get('uploaded_img_file_path', None)
    output_image_path = detect_object(uploaded_image_path)
    print(output_image_path)
    return render_template('show_image.html', user_image = output_image_path)
 
# flask clear browser cache (disable cache)
# Solve flask cache images issue
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response
 
if __name__=='__main__':
    app.run(debug = True)