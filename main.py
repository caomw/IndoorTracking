import cv2
import numpy as np

#from matplotlib import pyplot as plt

#####
## tools
def initCap():
    return cv2.VideoCapture('data/cam131.avi')
def gray(im):
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
def diff_abs(prev,cframe,fframe):
    abs_diff1 = cv2.absdiff(gray(prev),gray(fframe))
    abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
    return cv2.bitwise_and(abs_diff1,abs_diff2)
def diff_abs1(prev,cframe):
    return cv2.absdiff(gray(prev),gray(cframe))
def boundingBox(thresh,canvas):
    # find bounding rectangle
    ####
    # get non-zero indices
    nzx,nzy = np.nonzero(thresh)
    if any(nzx) and any(nzy):
        pt1 = (np.min(nzy),np.min(nzx))
        pt2 = (np.max(nzy),np.max(nzx))
        cv2.rectangle(canvas,pt1,pt2,(0,255,0),2)
        return canvas,thresh[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    else:
        return canvas,thresh

def plotHP(thresh):
    plt.plot(np.sum(thresh,axis=0))
    plt.show()

   
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
        #   diff = diff_abs(prev,cframe,fframe)
        diff = diff_abs1(prev,cframe)

        # thresholding
        _,thresh = cv2.threshold(diff,50,255,cv2.THRESH_BINARY)

        bbox,roi = boundingBox(thresh,cframe.copy())
        
        cv2.imshow('Temporal Difference',diff)
        cv2.imshow('Threshold',thresh)
        cv2.imshow('BoundingBox',bbox)
        cv2.imshow('ROI',roi)
        cv2.waitKey(40)

        prev = cframe

cv2.destroyAllWindows()
