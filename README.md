# Visual Odometry Midterm Project
automatic stitching of two images

## ORB
compute ORB keypoint and descriptors + apply Bruteforce matching with Hamming distance
opencv 함수를 사용하여 ORB keypoint와 descriptor를 구한 뒤 hamming distance를 토대로 bruteforce keypoint matching을 진행한다.

## RANSAC + Homography
implement RANSAC algorithm to compute the homography matrix
최적의 homography 행렬을 찾은 후 이를 통해 img2의 특정 좌표들을 특정 좌표로 변환하고
img2에 적용하여 이미지 전체를 특정 좌표로 변환한다.
이후 img1과 img2의 matching된 keypoint를 통해 이미지를 stitching하여 파노라마 이미지를 생성한다.
