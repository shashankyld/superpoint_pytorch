import os 
import sys
import numpy as np
import cv2

img_path = "../data/downloaded/indoor.jpg"
image = cv2.imread(img_path)
# Resize to max width or height of 240 pixels
max_dim = 500
height, width = image.shape[:2]
scaling_factor = max_dim / float(max(height, width))
image = cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detecting corners
corners = cv2.goodFeaturesToTrack(
    gray_image, maxCorners=1000, 
    qualityLevel=0.1, minDistance=5
    )

if corners is not None: 
    for i in corners: 
        x,y = i.ravel() # Ravel flattens the array
        cv2.circle(image, (int(x),int(y)), 3, 255, -1)

#Display

cv2.imshow("Image with good features to track", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save if file doesn't exist
output_path = img_path.replace(img_path.split('/')[-1], img_path.split('/')[-1].split('.')[0] + '_goodfeaturestotrack.jpg')

if not os.path.exists(output_path):
    cv2.imwrite(output_path, image)
    print("File saved at :", output_path)