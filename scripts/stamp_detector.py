import numpy as np
import imutils
import cv2


img = cv2.imread('../imgs/img.jpg')
img = imutils.resize(img, width=500)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

cv2.imshow('Original image', img)
cv2.imshow('Gray', gray)
cv2.imshow('Thresh', thresh)

cv2.waitKey(0)
cv2.destroyAllWindows()
