import cv2
import numpy as np
import os
from cv2 import xfeatures2d
from matplotlib import pyplot as plt

path = 'Basis Image'

orb = cv2.ORB_create(nfeatures=1000)

#-- this code serves as an array the images and the classNames --
images = []
classNames = []
myBasisImageList = os.listdir(path)

#--this portion is to count the number of images inside my basis image folder--
#print('Total Basis Images', len(myBasisImageList))
for cl in myBasisImageList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    #-- this line of code is where we add value to our array images by appending all the images in our basis folder --
    images.append(imgCur)
    #--when we print cl like this print(cl) it will display the images names with extensions -- 
    #-- in order for you to remove it you need this code os.path.splitext(cl)[0] --
    classNames.append(os.path.splitext(cl)[0])
#-- inorder for you to print the names you need to print the classNames, because the classNames contains the images names --
#print(classNames)

#--this is a function and we are passing the basis images that we have on it --
def findDes(images):
    # -- this variable is where we will store all of the image in the basis folder
    desList=[]
    #-- this for loop, it loops all the images in the image basis folder and get its keypoints --
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

# -- this variable will be calling the findDes function by puting the basis images into its parameter so that the return value is the length
# of how many basis images we have --
desList = findDes(images)

#--this is the captured image --
capturedImageResult1 = cv2.imread('Captured Image/captured.jpg')
capturedImageResult2 = cv2.imread('Captured Image/captured.jpg')
capturedImageResult3 = cv2.imread('Captured Image/captured.jpg')
capturedImageResult4 = cv2.imread('Captured Image/captured.jpg')



#------This block of code will display the captured image in black and white and in original 
#------ with red bounding box for height and width --------------------

gray = cv2.cvtColor(capturedImageResult2, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(gray,(3,3))

img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

img_log = np.array(img_log, dtype=np.uint8)

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

edges = cv2.Canny(bilateral, 100, 150)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
img_contours = cv2.drawContours(capturedImageResult2, contours, -1, (0, 0, 255), 2)

# Measure the height and width of each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)

kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg = cv2.drawKeypoints(closing, keypoints, None)



# Find largest contour (assumed to be the crack)
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding rectangle for largest contour
x, y, w, h = cv2.boundingRect(largest_contour)

# Draw rectangle on image
cv2.rectangle(capturedImageResult2, (x, y), (x+w, y+h), (255, 0, 0), 2)
cv2.putText(capturedImageResult2, "{} x {}".format(w, h), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# Print width and height of crack
print("Width: ", w)
print("Height: ", h)
#---------------------------------------------------------




#------This block of code will display the captured image in original with color indication
#------ on crack --------------------

gray = cv2.cvtColor(capturedImageResult3, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(gray,(3,3))

img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

img_log = np.array(img_log, dtype=np.uint8)

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

edges = cv2.Canny(bilateral, 100, 150)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
img_contours = cv2.drawContours(capturedImageResult3, contours, -1, (0, 0, 255), 2)

kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg1 = cv2.drawKeypoints(closing, keypoints, None)


# Find largest contour (assumed to be the crack)
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding rectangle for largest contour
x, y, w, h = cv2.boundingRect(largest_contour)



# Print width and height of crack
print("Width: ", w)
print("Height: ", h)
#---------------------------------------------------------




#------This block of code will display the captured image in original 
#------ with red bounding box for height and width in all cracks--------------------

gray = cv2.cvtColor(capturedImageResult4, cv2.COLOR_BGR2GRAY)

blur = cv2.blur(gray,(3,3))

img_log = (np.log(blur+1)/(np.log(1+np.max(blur))))*255

img_log = np.array(img_log, dtype=np.uint8)

bilateral = cv2.bilateralFilter(img_log, 5, 75, 75)

edges = cv2.Canny(bilateral, 100, 150)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on original image
img_contours = cv2.drawContours(capturedImageResult4, contours, -1, (0, 0, 255), 2)

# Measure the height and width of each contour
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    # Draw rectangle on image
    cv2.putText(capturedImageResult4, "{} x {}".format(w, h), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img_contours = cv2.rectangle(img_contours, (x, y), (x + w, y + h), (255, 0, 0), 2)  

kernel = np.ones((5,5), np.uint8)
closing = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel) 

sift = cv2.xfeatures2d.SIFT_create()

keypoints, descriptors = sift.detectAndCompute(closing, None)
featuredImg2 = cv2.drawKeypoints(closing, keypoints, None)

# Find largest contour (assumed to be the crack)
largest_contour = max(contours, key=cv2.contourArea)

# Get bounding rectangle for largest contour
x, y, w, h = cv2.boundingRect(largest_contour)



# Print width and height of crack
print("Width: ", w)
print("Height: ", h)
#---------------------------------------------------------




#--this function is we will pass here the captured image and all the basis images for comparison --
def findID(capturedImageResult1, desList):
    kp2, des2 = orb.detectAndCompute(capturedImageResult1, None)
    bf = cv2.BFMatcher()
    #-- this array or list is where we will store all the value of matches in every comparison that it did --
    matchList = []
    #-- this variable will be used to store the best match image with the highest key points --
    bestMatch = -1
    #-- this loop, it loops all the basis images inside or desList Array and then it will use to match for the captured image --
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance<0.75 * n.distance:
                    good.append([m])
            #print(len(good))
            matchList.append(len(good))
        #print(matchList)
    except:
        pass
    # -- this condition is to get for the maximum keypoints or the image who got the highest matches as long as it is not less than 15 because less than 15 is not a 
    #good match but in my case we will put 5 only for threshold value because we just need the highest one so that we can output or display an image --
    if len(matchList) != 0:
        if(max(matchList)) > 5:
           bestMatch = matchList.index(max(matchList))
    return bestMatch
           
#--this variable will store the index of that image who got the highest key points that matches the captured image --
id = findID(capturedImageResult1, desList)

# -- if the id == -1 then it means that the basis image folder is empty
if id != -1:
    #print(classNames[id])
    basisImageFolderPath = 'Basis Image'
    image_filenames = os.listdir(basisImageFolderPath)
    image_filename = image_filenames[id]
    
    BestMatchImageFromBasisImages = cv2.imread('Basis Image/' + image_filename)

    bestMatchBasisImage_new_shape = (400, 400)
    final_bestMatchBasisImage = cv2.resize(BestMatchImageFromBasisImages, bestMatchBasisImage_new_shape)
    capturedImage_new_shape = (400, 400)
    final_capturedImage = cv2.resize(capturedImageResult1, capturedImage_new_shape)
    
    final_good = []

    final_kp1, final_des1 = orb.detectAndCompute(final_bestMatchBasisImage, None)
    final_kp2, final_des2 = orb.detectAndCompute(final_capturedImage, None)
    
    final_bf = cv2.BFMatcher()
    final_matches = final_bf.knnMatch(final_des1, final_des2, k=2)
    
    for m, n in final_matches:
        if m.distance<0.75 * n.distance:
            final_good.append([m])

    
    
    displayImage = cv2.drawMatchesKnn(final_capturedImage, final_kp1, final_bestMatchBasisImage, final_kp2, final_good, None, flags=2)
    cv2.imshow('Result 1', displayImage)
    cv2.imshow('Result 2', capturedImageResult2)
    cv2.imshow('Result 3', featuredImg)
    cv2.imshow('Result 4', capturedImageResult3)
    cv2.imshow('Result 5', capturedImageResult4)

    #plt.subplot(121), plt.imshow(capturedImageResult3)
    #plt.title('Original'), plt.xticks([]), plt.yticks([])
    #plt.subplot(122), plt.imshow(featuredImg, cmap='gray')
    #plt.title('Output Image'), plt.xticks([]), plt.yticks([])
    #plt.show()

