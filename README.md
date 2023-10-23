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

### 2 images used for stitching
![img3_1](https://github.com/topia1998/Visual_Odometry/assets/88184675/4762feb4-d2a7-456d-ad83-6385b574ef39)
![img3_2](https://github.com/topia1998/Visual_Odometry/assets/88184675/8d041663-164b-431f-8f98-d76de7b9f191)
### result
![img3_result](https://github.com/topia1998/Visual_Odometry/assets/88184675/0a9dda39-b434-4e03-97e3-4c0707490b06)

### 2 images used for stitching
![img5_1](https://github.com/topia1998/Visual_Odometry/assets/88184675/20d8ad1a-754f-4a82-bc1b-b09798497062)
![img5_2](https://github.com/topia1998/Visual_Odometry/assets/88184675/af5d0336-ab89-41ce-8264-dd7869bdb897)
### result
![img5_result](https://github.com/topia1998/Visual_Odometry/assets/88184675/cdbbac82-0c10-416e-9036-349df867785c)
