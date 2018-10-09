import cv2
import os
import numpy as np
import pydicom
import math

class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def length(self):
        return numpy.sqrt(pow(self.x2 - self.x1, 2) + pow(self.y2 - self.y1, 2))

    def angle(self):
        return math.degrees(math.atan2(self.y2 - self.y1, self.x2 - self.x1))

def find_lines(input):
    """Finds all line segments in an image.
    Args:
        input: A numpy.ndarray.
    Returns:
        A filtered list of Lines.
    """
    detector = cv2.createLineSegmentDetector()
    if (len(input.shape) == 2 or input.shape[2] == 1):
        lines = detector.detect(input)
    else:
        tmp = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        lines = detector.detect(tmp)
    output = []
    if len(lines) != 0:
        for i in range(1, len(lines[0])):
            tmp = Line(lines[0][i, 0][0], lines[0][i, 0][1],
                        lines[0][i, 0][2], lines[0][i, 0][3])
            output.append(tmp)
    return output


img = pydicom.dcmread('test.dcm').pixel_array
cv2.imshow("origional",img)

# img2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cv2.drawContours(img, contours, -1, (0,255,0), 3)


ddepth = cv2.CV_16S
scale = 1
delta = 0


lines = find_lines(grad)
dst = cv2.Canny(grad, 50, 200, None, 3)
cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
# Draw the lines
if lines is not None:
    for i in range(0, len(lines)):
        cv2.line(cdst, (lines[i].x1, lines[i].y1), (lines[i].x1, lines[i].y1), (0,0,255), 3, cv2.LINE_AA)
cv2.imshow("Sobel", grad)
cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)

cv2.waitKey(0)
cv2.destroyAllWindows()
