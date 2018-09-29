#contains code for reading dicom (hopefully) and sobol

import cv2
import os
import numpy as np
import pydicom
import math

img = pydicom.dcmread('test.dcm').pixel_array

# cv2.imshow('image',img)
# imgray = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
# ret, thresh = cv2.threshold(imgray, 127, 255, 0)
# img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,255,0), 3)
ddepth = cv2.CV_16S
scale =
delta = 0
grad_x = cv2.Sobel(img, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
# Gradient-Y
# grad_y = cv.Scharr(gray,ddepth,0,1)
grad_y = cv2.Sobel(img, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
abs_grad_x = cv2.convertScaleAbs(grad_x)
abs_grad_y = cv2.convertScaleAbs(grad_y)
grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
lines = find_lines(grad)
dst = cv2.Canny(grad, 50, 200, None, 3)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        cv2.line(cdst, (lines[i].x1, lines[i].y1), (lines[i].x1, lines[i].y1), (0,0,255), 3, cv2.LINE_AA)
cv2.imshow("Sobel", grad)
# cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

cv2.waitKey(0)
cv2.destroyAllWindows()
