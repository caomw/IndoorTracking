import numpy as np
import cv2

#from matplotlib import pyplot as plt

import hickle as hkl

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


im = cv2.imread('roi.png',0)
hpv = hp(im)
print hpv.shape

#hkl.dump(hpv,'hpv.hkl')

rsv = rs(hpv)

hkl.dump(rsv,'rsv.hkl')

