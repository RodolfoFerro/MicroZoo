import numpy as np
import imutils
import cv2


img = cv2.imread('../imgs/img.jpg')
img = imutils.resize(img, width=500)
cimg = img.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

THR_FILTER = cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
ret, thresh = cv2.threshold(gray, 0, 255, THR_FILTER)

cv2.imshow('Original image', img)
cv2.imshow('Gray', gray)
cv2.imshow('Thresh', thresh)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

# circles = np.uint16(np.around(circles))
# for i in circles[0, :]:
#     # draw the outer circle
#     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
#     # draw the center of the circle
#     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow('Circles', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()
