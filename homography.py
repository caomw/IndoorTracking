import cv2 
import numpy as np
import hickle as hkl

def clickEvent(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print 'Loca : ',x, y
        pts.append((x,y))
# load the image, clone it, and setup the mouse callback function
image = cv2.resize(cv2.imread('./data/w04.png',0),(0,0),fx=0.5,fy=0.5)
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", clickEvent)
pts = []

# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF
    corres_pts = np.array( [ (0,0), (image.shape[1],0),(image.shape[1],image.shape[0]),(0,image.shape[0]) ] )

    if len(pts) == 4:
        pts = np.array(pts)
        # loop over the points and draw them on the cloned image
        for (x, y) in pts:
            cv2.circle(clone, (x, y), 5, (0, 255, 0), -1)

        h, status = cv2.findHomography(pts.astype('float32'), corres_pts.astype('float32'))
        #h = np.asarray([ [  6.68659284e-01,  -3.18388096e+00,   2.04840450e+03], [ -2.96960778e+00,  -6.80892728e-01,   3.30788559e+03,], [ -3.07254161e-04,   1.65667761e-03,   1.00000000e+00]])
        print h
        hkl.dump({'h' : h},'.config')
        warped = cv2.warpPerspective(image, h, (image.shape[1],image.shape[0]) )
        # apply the four point tranform to obtain a "birds eye view" of
        # the notecard
        #warped = perspective.four_point_transform(image, pts)
        #cv2.
        cv2.imshow('warped',warped)
        key = cv2.waitKey(-1)
        cv2.imwrite('data/warped.png',warped)

        break

 
cv2.destroyAllWindows()
 
