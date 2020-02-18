# Image-Stitching
Image stitching using homography estimation and RANSAC
1. Choose one image as the reference frame.
2. Estimate homography between each of the remaining images and the reference image. To estimate homography between two images, we use the following procedure: (a) Detect local features in each image (using SIFT). (b) Extract feature descriptor for each feature point. (c) Match feature descriptors between two images. (d) Robustly estimate homography using RANSAC.
3. Warp each image into the reference frame and composite warped images into a single mosaic.
