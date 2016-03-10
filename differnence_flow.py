import cv2
import numpy as np

import hickle as hkl

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

        if np.sum(roi) > 255*200:
            return canvas,roi,True
        else: 
            return canvas,thresh,False
    else:
        return canvas,thresh,False

def getFlow(prvs,next,hsv):
    flow = cv2.calcOpticalFlowFarneback(prvs,next, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr
def warpView(image):
    h = config['h']
    #return cv2.warpPerspective(image, h, (image.shape[1],image.shape[0]) ) 
    return cv2.warpPerspective(image, h, (1000,1000) ) 

###
# ___main___
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
active,frame = cam_feed.read()
prev = cv2.resize(warpView(frame),(0,0),fx=0.5,fy=0.5)

# hsv for optical flow
hsv = np.zeros_like(prev)
hsv[...,1] = 255


###
# the loop
###
k =0
while active:
    active,frame = cam_feed.read()
    cframe = cv2.resize(warpView(frame),(0,0),fx=0.5,fy=0.5)
    if active:
        # get difference image
        #   diff = cv2.subtract(gray(frame),prev)
        diff = diff_abs1(prev,cframe)
        #   diff = diff_abs1(prev,cframe)
        # diff = diff_abs2(prev,cframe,fframe)

        # thresholding
        _,thresh = cv2.threshold(diff,40,255,cv2.THRESH_BINARY)

        bbox,roi,useful = boundingBox(thresh,cframe.copy())

        # calculate horizontal projection profile
        #   >> comment : seems to work fine in real time
        # hpv = hp(roi)
        # rsv = rs(hpv)
        if useful:
            # operate on the Region of Interest
            flow_im = getFlow(gray(prev),gray(cframe),hsv)
            cv2.imshow('Flow',flow_im)

        k = cv2.waitKey(10) & 0xff
        prev = cframe

    cv2.imshow('Warped',cframe)
    cv2.imshow('Src',frame)

    if k == 27:
        break
    elif k == ord('s'):
        #cv2.imwrite('opticalfb.png',frame2)
        cv2.imwrite('opticalhsv.png',bgr)
 
cam_feed.release()
cv2.destroyAllWindows()
