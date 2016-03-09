import cv2
import numpy

#####
## tools
def initCap():
    return cv2.VideoCapture('data/cam131.avi')
def gray(im):
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
####

###
# ___main___
###
cam_feed = initCap()
# get an image
active,_ = cam_feed.read()
prev = gray(_)
print prev.shape

###
# the loop
###
while active:
    active,frame = cam_feed.read()
    if active:
        cv2.imshow('CAMFEED',cv2.subtract(gray(frame),prev))
        cv2.waitKey(40)
        prev = gray(frame)

cv2.destroyAllWindows()
