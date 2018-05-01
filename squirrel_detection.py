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
import math
from picamera.array import PiRGBArray
from picamera import PiCamera
import RPi.GPIO as GPIO

# setup stepper motors
mh = Adafruit_MotorHAT(addr=0x6F)
st1 = threading.Thread()
st2 = threading.Thread()

#for distance calculations
vi=60
viy=vi*math.cos(math.pi/4)

# alpha: The background learning factor, between 0 and 1. For
# static background use lower value (0.001). If background
# has moving objects, use a higher value (0.01).
class BackGroundSubtractor:
    def __init__(self, alpha, firstFrame):
        self.alpha = alpha
        self.backGroundModel = firstFrame

    def getForeground(self, frame):
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

#Calculate theta based on coordinates
#theta is the horizontal angle
def get_theta(x, z):
    return math.degrees(math.atan(x/z))

#Calculate phi based on coordiantes using projectile motion
#phi is the vertical angle
def move_motors(theta, phi):
    global st1, st2, mh
    #initialize stepper motors
    myStepper1 = mh.getStepper(200, 1)      # 200 steps/rev, motor port #1
    myStepper2 = mh.getStepper(200, 2)      # 200 steps/rev, motor port #1
    myStepper1.setSpeed(60)                 # 30 RPM
    myStepper2.setSpeed(60)                 # 30 RPM
    position1 = 2*int(round(theta/1.8))
    position2 = 2*int(round(phi/1.8))
    #begin thread to get stepper motor 1 (theta) moving
    if not st1.isAlive():
        if position1> 0:
            direction1 = Adafruit_MotorHAT.FORWARD
        else:
            direction1 = Adafruit_MotorHAT.BACKWARD
        st1 = threading.Thread(target=stepper_worker, args=(myStepper1, abs(position1),direction1, Adafruit_MotorHAT.INTERLEAVE))
        st1.start()
    #begin thread to get stepper motor 2 (phi) moving at the same time as stepper motor 1
    if not st2.isAlive():
        if position2> 0:
            direction2 = Adafruit_MotorHAT.BACKWARD
        else:
            direction2 = Adafruit_MotorHAT.FORWARD
        st2 = threading.Thread(target=stepper_worker, args=(myStepper2, abs(position2), direction2, Adafruit_MotorHAT.INTERLEAVE))
        st2.start()
    time.sleep(0.1)
    #wait for stepper motors to finish moving
    while st1.isAlive() or st2.isAlive():
        time.sleep(0.1)
    #open valve
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(14, GPIO.OUT)
    GPIO.output(14, 1)
    time.sleep(.25)
    GPIO.output(14, 0)
    raw_input("check angle")
    #return stepper motors to origin
    if position1> 0:
        direction1 = Adafruit_MotorHAT.BACKWARD
    else:
        direction1 = Adafruit_MotorHAT.FORWARD
    st1 = threading.Thread(target=stepper_worker, args=(myStepper1, abs(position1), direction1, Adafruit_MotorHAT.INTERLEAVE))
    st1.start()
    if position2> 0:
        direction2 = Adafruit_MotorHAT.FORWARD
    else:
        direction2 = Adafruit_MotorHAT.BACKWARD
    st2 = threading.Thread(target=stepper_worker, args=(myStepper2, abs(position2), direction2, Adafruit_MotorHAT.INTERLEAVE))
    st2.start()
    time.sleep(0.1)
    while st1.isAlive() or st2.isAlive():
        time.sleep(0.1)
    #turn off motors to reduce power comsumption and enable easier recalibration
    turnOffMotors()
    GPIO.cleanup()
        
# function to turn off motors
def turnOffMotors():
    global mh
    mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
    mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

def main():
    # load the class labels and Caffe model
    rows = open("synset_words.txt").read().strip().split("\n")
    classes = [r[r.find(" ") + 1:].split(",")[0] for r in rows]
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "squirrel_alexnet.caffemodel")

    # initialize the camera and grab a reference to the raw camera capture
    camera = PiCamera()
    camera.resolution = (640,480)
    camera.framerate = 4
    rawCapture = PiRGBArray(camera, size=(640,480))
    camera.capture(rawCapture, format="bgr")
    camera.brightness=50
    image = rawCapture.array
    backSubtractor = BackGroundSubtractor(0.4, denoise(image))
    rawCapture.truncate(0)

    #loop where frame from video feed is analyzed
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        # Read a frame from the camera
        frame = frame.array

        # get the foreground
        foreGround = backSubtractor.getForeground(denoise(frame))

        # Apply thresholding on the background and display the resulting mask
        ret, mask = cv2.threshold(foreGround, 15, 255, cv2.THRESH_BINARY)
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        c = get_best_contour(gray, 400)

        #check if there is motion in the frame
        if c is not None:
            # compute the bounding box for the contour
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            xVal = str(x)
            yVal = str(y)

            # crop the image and send to model to obtain predictions
            crop_img = image[y:(y + h), x:(x + w)]
            blob = cv2.dnn.blobFromImage(cv2.resize(crop_img, (224, 224)), 1, (224, 224), (104, 117, 123))
            net.setInput(blob)
            preds = net.forward()
            idxs = np.argsort(preds[0])[::-1][:5]

            # display bounding box, with prediction, percentage, and (x, y) coordinates
            if(preds[0][0] >= 0):
                disp = "{}: {:.2f}%     ({}, {})".format(classes[idxs[0]], preds[0][idxs[0]] * 100, xVal, yVal)
                cv2.putText(frame, disp, (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0))

                # calculate x, y, and z in feet
                z_coor = float(math.sqrt(93550/(w*h)))
                x_coor = float(z_coor*(x-320)*math.tan(31.1*math.pi/180)/320)
                y_coor = float(z_coor*(240-y)*math.tan(24.4*math.pi/180)/240)
                max_z=(vi**2/64)*(1+math.sqrt(1+4*32*y_coor/vi**2))
                # move the stepper motors
                if z_coor != 0 and y_coor != 0 and z_coor<max_z and classes[idxs[0]]=="squirrel" and h<100 and w<100:
                    theta  = get_theta(x_coor, z_coor)
                    phi = get_phi(x_coor, y_coor, z_coor, 32)
                    print("phi")
                    print(phi)
                    print("theta")
                    print(theta)
                    move_motors(theta, phi)
        #show frame and get camera ready to take next picture
        cv2.imshow('frame',frame)
        rawCapture.truncate(0)

        # exit program Ctrl+C
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            break

if __name__ == "__main__":
	main()

cam.release()
cv2.destroyAllWindows()
atexit.register(turnOffMotors)
