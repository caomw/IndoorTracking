import cv2
import numpy as np

#from matplotlib import pyplot as plt

import scipy
from lib.peakdetect import peakdetect
from lib.smooth import smoothify

#####
## tools
def initCap():
    return cv2.VideoCapture('data/cam131.avi')
    #return cv2.VideoCapture('data/02.mp4')
def gray(im):
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
def diff_abs(prev,cframe,fframe):
    abs_diff1 = cv2.absdiff(gray(prev),gray(fframe))
    abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
    return cv2.bitwise_and(abs_diff1,abs_diff2)
def diff_abs1(prev,cframe):
    return cv2.absdiff(gray(prev),gray(cframe))
def diff_abs2(prev,cframe,fframe):
    abs_diff1 = cv2.absdiff(gray(prev),gray(cframe))
    abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
    return abs_diff1 + abs_diff2
def boundingBox(thresh,canvas):
    # find bounding rectangle
    ####
    # get non-zero indices
    nzx,nzy = np.nonzero(thresh)
    if any(nzx) and any(nzy):
        pt1 = (np.min(nzy),np.min(nzx))
        pt2 = (np.max(nzy),np.max(nzx))
        cv2.rectangle(canvas,pt1,pt2,(0,255,0),2)
        roi = thresh[pt1[1]:pt2[1],pt1[0]:pt2[0]]

        if np.sum(roi) > 255*1000:
            return canvas,roi,True
        else: 
            return canvas,thresh,False
    else:
        return canvas,thresh,False

def plotHP(thresh):
    plt.plot(np.sum(thresh,axis=0))
    plt.show()

def hp(im):
    return np.sum(im,axis=0)

def w(x,sigma=10):
    return np.exp( -np.square( x/sigma ) )

def rs(hpv):
    ind = np.arange(hpv.shape[0]).reshape(hpv.shape)
    rsv = []
    for x in range(hpv.shape[0]):
        rsv.append(np.sum( np.multiply(w((ind[:x])[::-1]),hpv[:x]) ) - np.sum( np.multiply(w((ind[x:])[::-1]),hpv[x:]) ))

    return np.asarray(rsv)

def verticalSegmentation(rsv,roi,canvas):
    # smooth curve
    #maxi,mini = peakdetect(scipy.signal.savgol_filter(rsv,55,3),lookahead=100)
    maxi,mini = peakdetect(smoothify(rsv),lookahead=120)

    for i in range(len(maxi)):
        if i < len(mini):
            # check each segment for information content 
            info_content = np.sum(roi[:,maxi[i][0]:mini[i][0]])
            #if  info_content> 20000:
            #    print info_content
            cv2.line(canvas,(int(mini[i][0]),0),(int(mini[i][0]),canvas.shape[0]),255,2)
            cv2.line(canvas,(int(maxi[i][0]),0),(int(maxi[i][0]),canvas.shape[0]),255,2)

    return (maxi,mini),canvas
    


   
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
        diff = diff_abs(prev,cframe,fframe)
        #   diff = diff_abs1(prev,cframe)
        # diff = diff_abs2(prev,cframe,fframe)

        # thresholding
        _,thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)

        bbox,roi,useful = boundingBox(thresh,cframe.copy())

        # calculate horizontal projection profile
        #   >> comment : seems to work fine in real time
        # hpv = hp(roi)
        # rsv = rs(hpv)
        if useful:
            segments,segmented = verticalSegmentation(rs(hp(roi)),roi,roi.copy())

        
        #cv2.imshow('Temporal Difference',diff)
        #cv2.imshow('Threshold',thresh)
        #cv2.imshow('BoundingBox',bbox)
        if useful:
            #cv2.imshow('ROI',roi)
            cv2.imshow('Segmented ROI',segmented)
            #cv2.waitKey(-1)
        cv2.waitKey(40)

        prev = cframe

cv2.destroyAllWindows()
