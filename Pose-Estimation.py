import cv2
import time
import matplotlib.pyplot as plt 
import numpy as np 



# Specify the paths for the 2 files
protoFile = "pose/coco/pose_deploy_linevec.prototxt"
weightsFile = "pose/coco/pose_iter_440000.caffemodel"
 
# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


cap = cv2.VideoCapture(0)


while True:


	_ , frame = cap.read()

	frameCopy = np.copy(frame)
	frameWidth = frame.shape[1]
	frameHeight = frame.shape[0]
	threshold = 0.1

	# Specify the input image dimensions
	inWidth = 200
	inHeight = 200

	POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

	
	# Prepare the frame to be fed to the network
	inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

	# Set the prepared object as the input blob of the network
	net.setInput(inpBlob)

	output = net.forward()

	H = output.shape[2]
	W = output.shape[3]
	# Empty list to store the detected keypoints
	points = []
	for i in range(18):
	    # confidence map of corresponding body's part.
	    probMap = output[0, i, :, :]
	 
	    # Find global maxima of the probMap.
	    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
	     
	    # Scale the point to fit on the original image
	    x = (frameWidth * point[0]) / W
	    y = (frameHeight * point[1]) / H
	 
	    if prob > threshold : 
	        cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
	        #cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, lineType=cv2.LINE_AA)
	 
	        # Add the point to the list if the probability is greater than the threshold
	        points.append((int(x), int(y)))
	    else :
	        points.append(None)

	for pair in POSE_PAIRS:
	    partA = pair[0]
	    partB = pair[1]

	    if points[partA] and points[partB]:
	    	cv2.line(frame, points[partA], points[partB], (0, 255, 255), 2)
	    	cv2.circle(frame, points[partA], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

 
	cv2.imshow("Output-Keypoints",frame)
	
	if cv2.waitKey(1)==32:
		break

cap.release()
cv2.destroyAllWindows()