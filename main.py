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
active,prev = cam_feed.read()

###
# the loop
###
while active:
    _,cframe = cam_feed.read()
    active,fframe = cam_feed.read()
    if active:
        # get difference image
        #   diff = cv2.subtract(gray(frame),prev)
        abs_diff1 = cv2.absdiff(gray(prev),gray(cframe))
        abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
        diff = cv2.bitwise_and(abs_diff1,abs_diff2)

        # thresholding
        _,thresh = cv2.threshold(abs_diff1,50,255,cv2.THRESH_BINARY)
        
        cv2.imshow('Temporal Difference',diff)
        cv2.imshow('Threshold',thresh)
        cv2.waitKey(40)

        prev = cframe

cv2.destroyAllWindows()
