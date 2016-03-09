import cv2
import numpy

#####
## tools
def initCap():
    return cv2.VideoCapture('../data/cam131.avi')
####

###
# ___main___
###
cam_feed = initCap()

###
# the loop
###
active = True
while active:
    active,frame = cam_feed.read()
    if active:
        imshow('CAMFEED',frame)
        cv2.waitKey(15)

cv2.destroyAllWindows()
