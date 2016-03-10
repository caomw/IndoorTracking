import cv2
import numpy as np

import hickle as hkl

'''
import scipy
from lib.peakdetect import peakdetect
from lib.smooth import smoothify
'''

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

def sliceImage(segments,im):
    # for each segment
    maxi,mini = segments
    im_segments = []
    if len(maxi) and len(mini):
        #print '_____'
        for i in range(len(maxi)):
            if i < len(mini):
                #print maxi[i][0],mini[i][0]
                im_segments.append(im[:,maxi[i][0]:mini[i][0]])
        #print 'e_____e'
    return im_segments

def candidates(im_segments):
    candidates = []
    for im_segment in im_segments:
        candidates.append(np.nonzero(im_segment))
    print len(candidates)
    return candidates
     
def saliency(pixel_candidates):
    npcand = np.asarray(pixel_candidates)
    print npcand.shape

def warpView(image):
    h = config['h']
    #return cv2.warpPerspective(image, h, (image.shape[1],image.shape[0]) ) 
    return cv2.warpPerspective(image, h, (1000,1000) ) 
   
####

###
# __name__ : ___main___
###

###
# load configuration
###
config = hkl.load('.config')

###
# get feed
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

        # thresholding + histogram equalization
        _,thresh = cv2.threshold(diff,15,255,cv2.THRESH_BINARY)
        thresh = cv2.equalizeHist(thresh)

        # noise removal
        #thresh = cv2.fastNlMeansDenoising(thresh)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        thresh = cv2.erode(thresh,element)
        thresh = cv2.dilate(thresh, None, iterations=15)
        #bbox,roi,useful = boundingBox(thresh,cframe.copy())

        # calculate horizontal projection profile
        #   >> comment : seems to work fine in real time
        # hpv = hp(roi)
        # rsv = rs(hpv)
        '''
        if useful:
            segments,segmented = verticalSegmentation(rs(hp(roi)),roi,roi.copy())
            im_segments = sliceImage(segments,roi.copy())
            if len(im_segments):
                pixel_candidates = candidates(im_segments)
                #saliency(pixel_candidates)
        '''

        
        #cv2.imshow('Temporal Difference',diff)
        cv2.imshow('Threshold',thresh)
        #cv2.imshow('BoundingBox',bbox)
        '''
        if useful:
            #cv2.imshow('ROI',roi)
            cv2.imshow('Segmented ROI',segmented)
            cv2.waitKey(-1)
        '''
        k = cv2.waitKey(40)
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('thresh.png',thresh)

        prev = cframe

cv2.destroyAllWindows()
