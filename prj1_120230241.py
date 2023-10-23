import cv2
import numpy as np

def draw_matches(img1, keypoints1, img2, keypoints2, matches):
    r, c = img1.shape[:2]
    r1, c1 = img2.shape[:2]

    output_img = np.zeros((max([r, r1]), c+c1, 3), dtype='uint8')
    output_img[:r, :c, :] = np.dstack([img1, img1, img1])
    output_img[:r1, c:c+c1, :] = np.dstack([img2, img2, img2])

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = keypoints1[img1_idx].pt
        (x2, y2) = keypoints2[img2_idx].pt

        cv2.circle(output_img, (int(x1),int(y1)), 4, (0, 255, 255), 1)
        cv2.circle(output_img, (int(x2)+c,int(y2)), 4, (0, 255, 255), 1)

        cv2.line(output_img, (int(x1),int(y1)), (int(x2)+c,int(y2)), (0, 255, 255), 1)
        
    return output_img

def apply_homography(H, points):
    num_points = len(points)
    transformed_points = np.zeros((num_points, 2), dtype=np.float32)
    
    for i in range(num_points):
        x, y = points[i]
        z = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        x_dst = (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / z
        y_dst = (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / z
        transformed_points[i] = [x_dst, y_dst]
    
    return transformed_points

def warpCustom(img, H, output_shape):
    rows, cols = img.shape[:2]
    output_img = np.zeros(output_shape, dtype=img.dtype)

    H_inv = np.linalg.inv(H)

    for y in range(output_shape[0]):
        for x in range(output_shape[1]):
            p_dst = np.array([x, y, 1])
            p_src = np.dot(H_inv, p_dst)
            p_src /= p_src[2]

            x_src, y_src = int(p_src[0]), int(p_src[1])

            if 0 <= x_src < cols and 0 <= y_src < rows:
                output_img[y, x] = img[y_src, x_src]

    return output_img

def warpImages(img1, img2, H):

    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]])
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]])

    list_of_points_2 = apply_homography(H, temp_points)

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_shape = (y_max - y_min, x_max - x_min, 3)
    output_img = warpCustom(img2, np.dot(H_translation, H), output_shape)
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1
        
    return output_img


def findHomographyRANSAC(src_pts, dst_pts, max_dist=4.0, max_iterations=2000, confidence=0.99):
    best_H = None
    best_inliers = 0
    src_pts = np.float32(src_pts)
    dst_pts = np.float32(dst_pts)
    num_points = src_pts.shape[0]
    
    for _ in range(max_iterations):
        random_indices = np.random.choice(num_points, 4, replace=False)
        random_src = src_pts[random_indices]
        random_dst = dst_pts[random_indices]

        A = []
        for i in range(4):
            x, y = random_src[i]
            u, v = random_dst[i]
            A.append([-x, -y, -1, 0, 0, 0, x * u, y * u, u])
            A.append([0, 0, 0, -x, -y, -1, x * v, y * v, v])
        A = np.array(A)
        U, S, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]
        
        transformed_pts = np.dot(H, np.vstack((src_pts.T, np.ones(num_points))))
        transformed_pts = transformed_pts[:2] / transformed_pts[2]
        
        errors = np.sqrt(np.sum((transformed_pts - dst_pts.T) ** 2, axis=0))      
        inliers = np.sum(errors < max_dist)
        
        if inliers > best_inliers:
            best_inliers = inliers
            best_H = H
        
        inlier_ratio = inliers / num_points
        if inlier_ratio >= confidence:
            break
    
    return best_H

def main():
    img1 = cv2.imread('img5_1.jpg')
    img2 = cv2.imread('img5_2.jpg')

    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)


    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    matches = bf.knnMatch(descriptors1, descriptors2,k=2)

    all_matches = []
    for m, n in matches:
        all_matches.append(m)

    img3 = draw_matches(img1_gray, keypoints1, img2_gray, keypoints2, all_matches[:50])
    cv2.imwrite('img5_key.jpg', img3)

    good = []
    for m, n in matches:
        if m.distance < 0.5 * n.distance:
            good.append(m)

    MIN_MATCH_COUNT = 10

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good])
        M = findHomographyRANSAC(src_pts, dst_pts, max_dist=4.0)
        
        if M is not None:
            result = warpImages(img2, img1, M)
            cv2.imwrite('img5_result.jpg', result)
        else:
            print("Not enough inliers to stitch images.")
    else:
        print("Not enough good matches to stitch images.")
        
if __name__ == "__main__":
    main()
