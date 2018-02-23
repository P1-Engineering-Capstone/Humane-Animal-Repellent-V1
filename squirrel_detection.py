from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import time
import imutils
import thread
import threading
import cv2
import atexit

# alpha: The background learning factor, between 0 and 1. For
# static background use lower value (0.001). If background
# has moving objects, use a higher value (0.01).
class BackGroundSubtractor:
	def __init__(self, alpha, firstFrame):
		self.alpha = alpha
		self.backGroundModel = firstFrame

	def getForeground(self,frame):
		# apply the background averaging formula:
		self.backGroundModel = frame * self.alpha + self.backGroundModel * (1 - self.alpha)
		return cv2.absdiff(self.backGroundModel.astype(np.uint8),frame)

# Denoise image before further processing
def denoise(frame):
    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    return frame

# Return the biggest contour in the image
def get_best_contour(imgmask, threshold):
    image, contours, heirarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best_area = threshold
    best_cnt = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > best_area:
            best_area = area
            best_cnt = cnt
    return best_cnt

def main():
	# cam = VideoStream(src=0).start()
	cam = cv2.VideoCapture("squirrel.mp4")

	# if VideoStream no ret
	ret, frame = cam.read()
	backSubtractor = BackGroundSubtractor(0.1, denoise(frame))

	# load the class labels
	rows = open("synset_words.txt").read().strip().split("\n")
	classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
	COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

	# load the caffe model
	print("[INFO] loading model...")
	net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "squirrel_alexnet.caffemodel")

	while(True):
		# Read a frame from the camera
		ret, frame = cam.read()

		# Show the filtered image
		# cv2.imshow('input', denoise(frame))

		# get the foreground
		foreGround = backSubtractor.getForeground(denoise(frame))

		# Apply thresholding on the background and display the resulting mask
		ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
		gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
		c = get_best_contour(gray, 400)

		if c is not None:
	        # compute the bounding box for the contour
			(x, y, w, h) = cv2.boundingRect(c)
			cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 255, 0), 2)
			xVal = str(x)
			yVal = str(y)

			# crop the image and send to model to obtain predictions
			crop_img = frame[y:(y + h), x:(x + w)]
			blob = cv2.dnn.blobFromImage(cv2.resize(crop_img, (224, 224)), 1, (224, 224), (104, 117, 123))
			net.setInput(blob)
			preds = net.forward()
			idxs = np.argsort(preds[0])[::-1][:5]

			# display bounding box, with prediction, percentage, and (x, y) coordinates
			disp = "{}: {:.2f}%     ({}, {})".format(classes[idxs[0]], preds[0][idxs[0]] * 100, xVal, yVal)
			cv2.putText(mask, disp, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255))

		cv2.imshow('mask',mask)

		# exit program Ctrl+C
		key = cv2.waitKey(10) & 0xFF
		if key == 27:
			break

if __name__ == "__main__":
	main()

cam.release()
cv2.destroyAllWindows()


'''
# create a default object, no changes to I2C address or frequency
 mh = Adafruit_MotorHAT()

# create empty threads (these will hold the stepper 1 and 2 threads)
st1 = threading.Thread()
st2 = threading.Thread()

# turn off the stepper motors at program exit
def turnOffMotors():
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

atexit.register(turnOffMotors)

myStepper1 = mh.getStepper(200, 1)      # 200 steps/rev, motor port #1
myStepper2 = mh.getStepper(200, 2)      # 200 steps/rev, motor port #1
myStepper1.setSpeed(60)                 # 30 RPM
myStepper2.setSpeed(60)                 # 30 RPM


stepstyles = [Adafruit_MotorHAT.SINGLE, Adafruit_MotorHAT.DOUBLE, Adafruit_MotorHAT.INTERLEAVE, Adafruit_MotorHAT.MICROSTEP]

def stepper_worker(stepper, numsteps, direction, style):
    stepper.step(numsteps, direction, style)
'''