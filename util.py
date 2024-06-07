import cv2
import math
import numpy as np

def get_polygon_mask(shape, vertices):
    mask = np.zeros(shape, dtype=np.uint8)
    # 다각형 영역 채우기
    cv2.fillPoly(mask, vertices, 255)

    return mask

def line_histogram_image(hist_img, position, color):
    hist_img[:, position] = color

    return hist_img

def get_histogram_image(hist, bin=256, height=256):        
    # 히스토그램을 이미지로 변환
    hist_img = np.zeros((height, bin, 3), dtype=np.uint8)

    # 히스토그램 정규화
    cv2.normalize(hist, hist, alpha=0, beta=hist_img.shape[0], norm_type=cv2.NORM_MINMAX)

    # 히스토그램을 이미지에 그리기
    for x in range(bin):
        cv2.line(hist_img, (x, height), (x, height - int(hist[x])), (255, 255, 255), 1)

    return hist_img

def rotate_image(src, degree):
    h, w = src.shape[:2]
    center = (w / 2, h / 2)

    rot = cv2.getRotationMatrix2D(center, degree, 1)

    rad = math.radians(degree)
    sin = math.sin(rad)
    cos = math.cos(rad)
    b_w = int((h * abs(sin)) + (w * abs(cos)))
    b_h = int((h * abs(cos)) + (w * abs(sin)))

    rot[0, 2] += ((b_w / 2) - center[0])
    rot[1, 2] += ((b_h / 2) - center[1])

    outImg = cv2.warpAffine(src, rot, (b_w, b_h), flags=cv2.INTER_LINEAR)
    
    return outImg