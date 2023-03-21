import cv2
import numpy as np

# Load the image and convert to grayscale
img = cv2.imread('Captured Image/captured.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(gray,(3,3))

img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

img_log = np.array(img_log, dtype=np.uint8)

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

edges = cv2.Canny(bilateral, 100, 150)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
img_contours = cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg1 = cv2.drawKeypoints(closing, keypoints, None)


# Find largest contour (assumed to be the crack)
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding rectangle for largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw a straight line on the longest crack
leftmost = tuple(largest_contour[largest_contour[:,:,0].argmin()][0])
rightmost = tuple(largest_contour[largest_contour[:,:,0].argmax()][0])
cv2.line(img, leftmost, rightmost, (0,255,0), 2)

# Print width and height of crack
print("Width: ", w)
print("Height: ", h)

# Display the result
cv2.imshow('Crack detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
