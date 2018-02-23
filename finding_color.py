import cv2
import numpy as np

#load picture
frame = cv2.imread('blue.jpeg')

# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of  color in HSV
#these ranges are for blue
lower_blue = np.array([100 ,50,50])
upper_blue = np.array([140,255,255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

cv2.imshow('frame',frame)
cv2.imshow('mask',mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
