import cv2
import numpy as np
import math

import hickle as hkl
import argparse 
import sys

#####
## tools
def initCap():
    if args['input']:
        return cv2.VideoCapture(args['input'])
    ## enable webcam
    return cv2.VideoCapture(0)

# grayscale
def gray(im):
    return cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

# get absolute differnce between
#   (prev,future) and (current,future)
#       detect large changes only
def diff_abs(prev,cframe,fframe):
    abs_diff1 = cv2.absdiff(gray(prev),gray(fframe))
    abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
    return cv2.bitwise_and(abs_diff1,abs_diff2)

# (prev,current) : keep most of the information
#   detects more changes
def diff_abs1(prev,cframe):
    return cv2.absdiff(gray(prev),gray(cframe))

# preserve even more information content
def diff_abs2(prev,cframe,fframe):
    abs_diff1 = cv2.absdiff(gray(prev),gray(cframe))
    abs_diff2 = cv2.absdiff(gray(cframe),gray(fframe))
    return abs_diff1 + abs_diff2

# centroid of points in a list of points [contour]
def centroid(contour):
    M = cv2.moments(contour)
    return int(M['m10']/M['m00']),int(M['m01']/M['m00'])

# Eucledian distance between points
def dist(p1,p2):
    return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )

# Check if two contours should "belong together"
def find_if_close(cnt1,cnt2):
    return ( dist(centroid(cnt1) , centroid(cnt2)) < 200 )

# Merge contours
def merge(cnts):
    hulls = []
    cnts_len = len(cnts)
    for i in range(cnts_len):
        for j in range(cnts_len):
            if find_if_close(cnts[i],cnts[j]):
                hulls.append(cv2.convexHull(cnts[i],cnts[j]))
    return hulls

# Get the coordinates of the bottom part of each contour
#   This is highly application specific
def getCoordinates(cnts,canvas):
    pts = []
    if len(cnts) < 1:
        return pts,canvas
    for cnt in cnts:
        npcnt = np.asarray(cnt).reshape([len(cnt),2]).T
        pt = tuple(npcnt.T[np.argmax(npcnt[1])])
        pts.append(pt)
        cv2.circle(canvas,pt,5,(255,0,0),-1)
    return pts,canvas

# Warp perspective to Top View
#   using transformation matrix 'h'
#       Set size of top view here : location/camera-orientation specific
def warpView(image,h,size=(1000,1000)):
    if args['topview']:
        size = tuple( [int(i) for i in args['topview'].split('x')] )
    return cv2.warpPerspective(image, h, size)

# Mat multiplication of ( h x point ) gives
#   the location of point in Top View
#       normalization of point to be in form
#          h x [x1,y1,1] = [x2,y2,1] 
def tranformCoordinates(pts,canvas):
    tpts = []
    if len(pts) > 0:
        for pt in pts:
            tpt = np.dot(h,(pt[0],pt[1],1))
            tpt = (tpt/tpt[2]).astype('int32')
            tpts.append(tpt)
            # negative values are a possibility
            dpt = ( max(0,tpt[0]),max(0,tpt[1]) )
            cv2.circle(canvas,dpt,15,(0,0,255),-1)
    return tpts,canvas
    
def _build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", help="A video file as input")
    ap.add_argument("-t", "--threshold", help="Set a threshold value once, which will be stored persistently")
    ap.add_argument("-d", "--cam",help="Cam device number : typically 0")
    ap.add_argument("-c", "--configure",help="Boolean : Configure transformation matrix by homography")
    ap.add_argument("-l", "--dilate",help="Dilation iterations count: (1-25)")
    ap.add_argument("-v", "--topview",help="Dimensions of Top View : a,b ")
    args = vars(ap.parse_args())

    return args

def clickEvent(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Loca : ',x, y
        pts.append((x,y))

def config_loop():
    global pts 
    active,image = cam_feed.read()

    while active:
        cv2.imshow('Perspective',image)
        key = cv2.waitKey(1) & 0xFF
        corres_pts = np.array( [ (0,0), (image.shape[1],0),(image.shape[1],image.shape[0]),(0,image.shape[0]) ] )

        if len(pts) == 4:
            pts = np.array(pts)
            for (x, y) in pts:
                cv2.circle(image, (x, y), 5, (0, 255, 0), -1)

            h, _ = cv2.findHomography(pts.astype('float32'), corres_pts.astype('float32'))
            print 'Tranformation Matrix : ',h
            # save to config file
            hkl.dump({'h' : h},'.config')
            # warp image
            warped = cv2.warpPerspective(image, h, (image.shape[1],image.shape[0]) )
            cv2.imshow('Perspective',image)
            cv2.imshow('warped',warped)
            # pause
            cv2.waitKey(-1)
            break
 

###
# __name__ : ___main___
###

###
# Parse Arguments
args = _build_args()

###
# get feed
cam_feed = initCap()

pts = []

###
# load configuration
if args['configure']:
    cv2.namedWindow("Perspective")
    cv2.setMouseCallback("Perspective",clickEvent)
    config_loop()
    print 'Configuration Done. Now run the program without --configure option'
    sys.exit()
else:
    config = hkl.load('.config')
    # configuration
    config = hkl.load('.config')
    # homography transformation matrix
    h = config['h']
    # get an image
    active,prev = cam_feed.read()
    # Get warp view
    warped = warpView(prev,h)
    cv2.imshow('warped',warped)


# delay for the loop
delay = 40

###
# the loop
###
while active:
    _,cframe = cam_feed.read()
    active,fframe = cam_feed.read()
    if active:
        # get difference image
        diff = diff_abs(prev,cframe,fframe)

        # thresholding + histogram equalization
        #   threshold 'th' => heuristic
        th = 15
        if args['threshold']:
            th = int(args['threshold'])
        _,thresh = cv2.threshold(diff,th,255,cv2.THRESH_BINARY)
        thresh = cv2.equalizeHist(thresh)

        # noise removal
        #thresh = cv2.fastNlMeansDenoising(thresh)
        # >> comment : denoising is too slow; ironically named 'fast*'

        ###
        # erosion for noise removal
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
        thresh = cv2.erode(thresh,element)
        # dilation
        #   10 -> heuristic
        dil = 10
        if args['dilate']:
            dil = int(args['dilate'])
        thresh = cv2.dilate(thresh, None, iterations=dil)

        # operate on the threshold image
        #   gather contours
        cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 2)

        # merge contours that are closer, together
        #   using ConvexHull which is surprisingly fast
        hulls = merge(cnts)
        # remove small contours 
        #   area threshold is a heuristic, could be adjusted
        hulls_filtered = [x for x in hulls if cv2.contourArea(x) > 5000]

        # get a copy of current frame for drawing contours
        canvas = cframe.copy()
        cv2.drawContours(canvas,hulls_filtered,-1,(0,255,0),2)
        coord,canvas = getCoordinates(hulls_filtered,canvas)

        # transformed coordinates
        #   Transform coordinates from Perspective to Top View
        #       Mat product : h x coord
        tcoords,topview = tranformCoordinates(coord,warped.copy())

        # display Threshold, Canvas, Top View
        cv2.imshow('Threshold',thresh)
        cv2.imshow('Canvas',canvas)
        cv2.imshow('warped',topview)

        k = cv2.waitKey(delay)
        if k == 27:
            break
        elif k == ord('p'):
            cv2.waitKey(-1)
        elif k == ord('s'):
            cv2.imwrite('thresh.png',thresh)
            cv2.imwrite('canvas.png',canvas)
            cv2.imwrite('topview.png',topview)
        elif k == ord('+'):
            delay = min(200,delay+5)
        elif k == ord('-'):
            delay = max(1,delay-5)

        # last frame
        prev = cframe

cam_feed.release()
cv2.destroyAllWindows()
