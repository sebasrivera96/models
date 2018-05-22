"""
    Script name: transformLogo.py
    Author: Sebastian Rivera Gonzalez
    Objective: Given 11 pictures taken with a Samsung Galaxy S7, a complete 
    dataset of the Continental Logo wants to be built.
        1) The 1st SECTION of the code will be to take an image and modify it
        in various ways to create more images with different characteristics.
        The library used to manipulate the images is OpenCV 2.4.9.1.
        2) The 2nd SECTION will be to resize all the images to dimensions 
        480x320 pixels. The reason for this is that the training is being 
        carried out on a CPU and if the size of the images is bigger, the OS
        will simply kill the process.
"""
#----------------------------------LIBRARIES------------------------------------
import cv2
import os
import numpy as np
#-------------------------------------------------------------------------------

#----------------------------------1st SECTION----------------------------------
def rotateImage(img_name, angle, save):
    img = cv2.imread(img_name)
    rows, cols, _ = img.shape
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    dstImg = cv2.warpAffine(img, M, (cols, rows))
    if save:
        cv2.imwrite(img_name + '_' + str(angle) + '.jpg',dstImg)
    return dstImg

def createVariantsOfLogo():
    for fname in os.listdir("."):
        if fname.endswith(".jpg"):
            img = cv2.imread(fname)
            rows,cols,_ = img.shape
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(ACT_IMG+'_gray.jpg',gray_img)
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            cv2.imwrite(ACT_IMG+'_hsv.jpg',hsv_img)
            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            cv2.imwrite(ACT_IMG+'_yuv.jpg',yuv_img)
            blur = cv2.blur(img,(5,5))
            cv2.imwrite(ACT_IMG+'_blur.jpg',blur)
            kernel = np.ones((5,5),np.uint8)
            erosion = cv2.erode(img,kernel,iterations = 1)
            cv2.imwrite(ACT_IMG+'_erosion.jpg',erosion)
            dilation = cv2.dilate(img,kernel,iterations = 1)
            cv2.imwrite(ACT_IMG+'_dilation.jpg',dilation)
            dst = rotateImage(ACT_IMG+'.jpg',90,False)
            dst = rotateImage(ACT_IMG+'.jpg',180,False)
            dst = rotateImage(ACT_IMG+'.jpg',270,False)
            # NEW
            dst = rotateImage(ACT_IMG+'.jpg',45,False)
            dst = rotateImage(ACT_IMG+'.jpg',135,False)
            dst = rotateImage(ACT_IMG+'.jpg',225,False)
            # cv2.imshow('frame', dst)
            # cv2.waitKey(0)

        

#-------------------------------------------------------------------------------

#---------------------------------2nd SECTION-----------------------------------
def resizeImages():
    FINAL_ROWS = 320.0
    FINAL_COLUMNS = 480.0
    for fname in os.listdir("."):
        if fname.endswith(".jpg"):
            img = cv2.imread(fname)
            rows,cols,_ = img.shape
            FX = FINAL_ROWS/float(rows)
            FY = FINAL_COLUMNS/float(cols)
            img = cv2.resize(img, (0,0), fx=FY, fy=FX)
            cv2.imwrite(fname, img)
            print img.shape
            # print fname
#-------------------------------------------------------------------------------
img = cv2.imread("logo1.jpg")
def translateImage(x1,x2,img):
    rows,cols,_ = img.shape
    M = np.float32([[1,0,100],[0,1,50]])
    dst = cv2.warpAffine(img,M,(cols,rows))
    cv2.imshow("image",dst)
    cv2.waitKey(0)
    return dst
# translateImage(200,-100,img)
resizeImages()
