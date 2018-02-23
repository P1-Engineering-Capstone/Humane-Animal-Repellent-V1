# Squirrel Repelling Robot
We trained a computer model using Caffe Framework and are developing an application written in Python which is able to run on 
a Raspberry Pi. The model was trained on a dataset of over 6,000 images with 4 classes: squirrels, humans, dogs, and cats.

Using OpenCV 3.4, we use an adaptive background subtracting function which detects motion in the current frame relative to 
previous frames. Cropping the frame based on the largest contour, we pass it into the trained model and receieve the 
predictions in order of highest percentage to lowest. If the object moving in the frame is a squirrel, move the stepper motors 
to the correct corrdinates and activate the water nozzle to squirt water at the squirrel, scaring it away from people's gardens,
lawn pots, sprinklers, etc.

Find the model at: https://www.dropbox.com/s/n2qgpjm1slej0sf/squirrel_alexnet.caffemodel?dl=0

Details:
  Iteration
  Accuracy
  Loss
