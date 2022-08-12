import os, sys
import numpy as np
import cv2
import time
from imutils.object_detection import non_max_suppression

def east_detect(image):
    
    layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]
    
    orig = image.copy()
    result1 = image    
    
    if len(result1.shape) == 2:
        result1 = cv2.cvtColor(result1, cv2.COLOR_GRAY2RGB)
    
    (H, W) = result1.shape[:2]
    
    # set the new width and height and then determine the ratio in change
    # for both the width and height: Should be multiple of 32
    (newW, newH) = (320, 320)
    
    rW = W / float(newW)
    rH = H / float(newH)
    
    # resize the image and grab the new image dimensions
    result1 = cv2.resize(result1, (newW, newH))
    
    (H, W) = result1.shape[:2]
    
    net = cv2.dnn.readNet("/home/romain/EAST-Detector-for-text-detection-using-OpenCV/frozen_east_text_detection.pb")
    
    blob = cv2.dnn.blobFromImage(result1, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    start = time.time()
    
    net.setInput(blob)
    
    (scores, geometry) = net.forward(layerNames)
    
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []
    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]
    
        for x in range(0, numCols):
    		# if our score does not have sufficient probability, ignore it
            # Set minimum confidence as required
            if scoresData[x] < 0.0001:
                continue
    		# compute the offset factor as our resulting feature maps will
            #  x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])
                        
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
    	# scale the bounding box coordinates based on the respective
    	# ratios
    	startX = int(startX * rW)
    	startY = int(startY * rH)
    	endX = int(endX * rW)
    	endY = int(endY * rH)
    	# draw the bounding box on the image
    	cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 0, 0), -1)
    
    # convert to gray
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    
    # threshold and invert
    thresh = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    
    # apply morphology close
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)
    
    # get contours and filter to keep only small regions
    mask = np.zeros_like(gray, dtype=np.uint8)
    cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
    for c in cntrs:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(mask,[c],0,255,-1)
    
    # do inpainting
    result1 = cv2.inpaint(orig,mask,3,cv2.INPAINT_TELEA)
        
    print(time.time() - start)
        
    return result1
    
for poster in os.listdir("./posters"):
    image = cv2.imread("./posters/"+poster)
    if image is None:
        continue
    
    out_image = east_detect(image)
    
    cv2.imwrite("./posters_without_text/"+poster, out_image)