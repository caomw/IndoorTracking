# Detecting People in Cluttered Indoor Scenes


The task given is essentially a combination of two sub-problems, 1. Capturing the silhouettes of people present in the room and 2. Extracting enough information from the environment and the silhouettes of each person, to project the same on to the top view.

## Assumptions

	- Human beings are the only moving objects
	- The orientation of the camera is known


## Silhouette or Blob detection

I came across three different ways of doing this, with reasonable performance in real time. In all of them, motion tracking is the primary activity. Background subtraction yields great results but it is also very sensitive to noise. Optical flow is very accurate compared to other methods, but it is incredibly slow, to run in real time, unless run on a GPU with a good parallel algorithm. The best solution seems to be temporal differencing. I read a paper on it, Detecting People in Cluttered Indoor Scenes, and attempted to implement it. They use Tensor Voting for removing outliers, noise and also to assign a signature to each collection of points (contours) for classification (as human/objects). Tensor voting is a sophisticated technique which needs to be implemented in a parallel manner, in order use it in real time applications.

The algorithm I used is rather simple. Difference image is calculated from previous, current and next frames, which is binarized (threshold is a heuristic). Noise in the image are removed by morphological operations, eroding and dilating, following which contours are collected from the image. The contours that are close together are grouped together by finding the convex hull for each pairs of contours. This process provides very good contours, from which the bottom most point in the contour is marked a the coordinate of that moving object.



## Perspective Transform

I used homography to transform the perspective view from the camera to top view. This process is unfortunately too complex to be automated (use of markers in the scene). Instead I gathered 4 points ( a,b,c,d) from the scene, which represent the floor, to be seen from the top view. By transforming these four points to the end points of our top view [ 0,0),(w,0),(w,h),(0,h) ], we get the transformation matrix H. Using H, the view is warped to the top view. The size of the top view is environment specific and should be entered by the user. Watch the demo video I have attached, to see how I gather 4 points for homography.

The coordinates obtained from the first part, are transformed by matrix multiplying it with H. The multiplication should be of the form ( h x [x1,y1,1] = [x2,y2,1]  ). The new coordinated are then plotted on the warped view.

## Drawbacks
This implementation is simplistic, in many ways, starting with the fact that it can track only moving objects. Multiple people could be grouped into the same blob. A person can be put in two or more different blobs. The plotting of coordinates in the top view is noisy, which can be corrected by using moving average and clustering closer coordinates. Given enough time, these problems can be addressed with patience and solved and the accuracy of tracking can be improved greatly. 

