import numpy as np
import cv2

#from matplotlib import pyplot as plt

import hickle as hkl


# find bounding rectangle
def boundingBox(thresh,canvas):
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



im = cv2.imread('roi.png',0)
hpv = hp(im)
print hpv.shape

#hkl.dump(hpv,'hpv.hkl')

rsv = rs(hpv)

hkl.dump(rsv,'rsv.hkl')

