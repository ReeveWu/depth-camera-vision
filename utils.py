import cv2
import numpy as np

'''
**************************************
*                                    *
*           Pre-processing           *
*                                    *
**************************************
'''

def do_clahe(left_image, right_image):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    left_image_clahe = clahe.apply(left_image)
    right_image_clahe = clahe.apply(right_image)

    d = 9
    sigmaColor = 75
    sigmaSpace = 75

    left_image_bilateral = cv2.bilateralFilter(left_image_clahe, d, sigmaColor, sigmaSpace)
    right_image_bilateral = cv2.bilateralFilter(right_image_clahe, d, sigmaColor, sigmaSpace)

    return left_image_bilateral, right_image_bilateral


def do_canny(left_img, right_img, sigma=0.33):
    def apply_auto_canny(image, sigma):
        v = np.median(image)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, lower, upper)
        return edged
    
    edges_left = apply_auto_canny(left_img, sigma)
    edges_right = apply_auto_canny(right_img, sigma)

    edges_left = np.maximum(edges_left, left_img)
    edges_right = np.maximum(edges_right, right_img)

    return edges_left, edges_right


def do_medianBlur(left_img, right_img):
    left_img = cv2.medianBlur(left_img, ksize=5)
    right_img = cv2.medianBlur(right_img, ksize=5)
    return left_img, right_img

'''
**************************************
*                                    *
*          Post-processing           *
*                                    *
**************************************
'''

def dilate_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_image = cv2.dilate(image, kernel, iterations=iterations)
    
    return dilated_image


def erode_image(image, kernel_size=3, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded_image = cv2.erode(image, kernel, iterations=iterations)
    
    return eroded_image


'''
**************************************
*                                    *
*        Compute deep image          *
*                                    *
**************************************
'''

preprocess = {
    "Histogram Equalization (CLAHE)": do_clahe,
    "Edge Detection (Canny)": do_canny,
    "Median Blur": do_medianBlur
}

def compute(left_image, right_image, resize, preprocesssing, min_disparity, num_disparities, block_size, uniqueness_ratio, speckle_window_size, speckle_range, p1, p2, delate, erode):
    left_image = cv2.resize(left_image, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)
    right_image = cv2.resize(right_image, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    for x in preprocesssing:
        left_image, right_image = preprocess[x](left_image, right_image)

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        uniquenessRatio=uniqueness_ratio,
        speckleWindowSize=speckle_window_size,
        speckleRange=speckle_range,
        P1=p1,
        P2=p2
    )

    disparity_map = stereo.compute(left_image, right_image).astype(np.float32)
    disparity_visual = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if delate > 1:
        disparity_visual = dilate_image(disparity_visual, kernel_size=delate)
    if erode > 1:
        disparity_visual = erode_image(disparity_visual, kernel_size=erode)

    return disparity_visual