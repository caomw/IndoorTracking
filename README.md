# Detecting People in Cluttered Indoor Scenes

## Temporal Difference

- [x] Obtain differene image for motion tracking

## Thresholding on Difference Image

- [ ] Read Rosin's work
- [x] Binary thresholding of difference image 
- [x] Draw bounding box 

## Horizontal Projection Profile

- [x] Implement horizontal projection profile
- [x] Implement Region boundary saliency 
- [ ] Vertical segmentation 

## Saliency Assesment

- [ ] Noise removal and outlier elimination by Tensor Voting
- [ ] Calculate Saliency Tensor for each candidate points
- [ ] Saliency of point 'i' : lambda(max) - lamda(min)

## Saliency Completion

- [ ] Use all the points in the bounding box as candidate
- [ ] Fix missing candidates

## Head Detection

- [ ] Observe change in orientation along head and shoulder
- [ ] Vertical segmentation : x-component of tangents
- [ ] Candidate points above the neck : head position and size



