import numpy as np
import imutils
import cv2

# Set global:
THR_FILTER = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU

# Preprocess original image:
img_rgb = cv2.imread('../imgs/img.jpg')
img_rgb = imutils.resize(img_rgb, width=640)
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img_gray, 0, 255, THR_FILTER)
cv2.imshow('Binarized image', img_thresh)

for i in range(1, 6):
    # Preprocess template:
    tmp_rgb = cv2.imread('../imgs/template{}.png'.format(i))
    tmp_rgb = imutils.resize(tmp_rgb, width=72)
    tmp_gray = cv2.cvtColor(tmp_rgb, cv2.COLOR_BGR2GRAY)
    ret, tmp_thresh = cv2.threshold(tmp_gray, 0, 255, THR_FILTER)
    cv2.imshow('Binarized template {}'.format(i), tmp_thresh)
    w, h = tmp_gray.shape

    # Template matching
    res = cv2.matchTemplate(img_thresh, tmp_thresh, cv2.TM_CCOEFF_NORMED)
    threshold = 0.7
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

# cv2.imwrite('../imgs/res.png', img_rgb)
cv2.imshow('Result', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
