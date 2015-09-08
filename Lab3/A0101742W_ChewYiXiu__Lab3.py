import cv2
import os
import numpy as np
import math
from matplotlib import pyplot as plt

#turn image to grayscale
img = cv2.imread(os.getcwd() + "/Pictures/test3.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
#define sobel and prewitt kernels
sobelx_kernel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
sobely_kernel = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
prewittx_kernel = np.array([[-1, 0, 1],[-1, 0, 1],[-1, 0, 1]])
prewitty_kernel = np.array([[1, 1, 1],[0, 0, 0],[-1, -1, -1]])

rows = img.shape[0]
columns = img.shape[1]
sobel_result = np.zeros([rows,columns]) 
prewitt_result = np.zeros([rows,columns])
maxsobelEdge = 0
maxprewittEdge = 0
for r in range(0,rows-2):
    for c in range(0,columns-2):
        sobelx = sobely = prewittx = prewitty = 0.0
        #convolve image with sobel and prewitt kernels
        sobelx = (img[r][c]*sobelx_kernel[0][0]) + (img[r][c+1]*sobelx_kernel[0][1]) + (img[r][c+2]*sobelx_kernel[0][2]) + (img[r+1][c]*sobelx_kernel[1][0]) + (img[r+1][c+1]*sobelx_kernel[1][1]) + (img[r+1][c+2]*sobelx_kernel[1][2]) + (img[r+2][c]*sobelx_kernel[2][0]) + (img[r+2][c+1]*sobelx_kernel[2][1]) + (img[r+2][c+2]*sobelx_kernel[2][2])

        sobely = (img[r][c]*sobely_kernel[0][0]) + (img[r][c+1]*sobely_kernel[0][1]) + (img[r][c+2]*sobely_kernel[0][2]) + (img[r+1][c]*sobely_kernel[1][0]) + (img[r+1][c+1]*sobely_kernel[1][1]) + (img[r+1][c+2]*sobely_kernel[1][2]) + (img[r+2,][c]*sobely_kernel[2][0]) + (img[r+2][c+1]*sobely_kernel[2][1])+ (img[r+2][c+2]*sobely_kernel[2][2])

        prewittx = (img[r][c]*prewittx_kernel[0][0]) + (img[r][c+1]*prewittx_kernel[0][1]) + (img[r][c+2]*prewittx_kernel[0][2]) + (img[r+1][c]*prewittx_kernel[1][0]) + (img[r+1][c+1]*prewittx_kernel[1][1]) + (img[r+1][c+2]*prewittx_kernel[1][2]) + (img[r+2][c]*prewittx_kernel[2][0]) + (img[r+2][c+1]*prewittx_kernel[2][1]) + (img[r+2][c+2]*prewittx_kernel[2][2])

        prewitty = (img[r][c]*prewitty_kernel[0][0]) + (img[r][c+1]*prewitty_kernel[0][1]) + (img[r][c+2]*prewitty_kernel[0][2]) + (img[r+1][c]*prewitty_kernel[1][0]) + (img[r+1][c+1]*prewitty_kernel[1][1]) + (img[r+1][c+2]*prewitty_kernel[1][2]) + (img[r+2][c]*prewitty_kernel[2][0]) + (img[r+2][c+1]*prewitty_kernel[2][1]) + (img[r+2,c+2]*prewitty_kernel[2][2])

        sobel_result[r+1][c+1] = math.sqrt(sobelx**2 + sobely**2)
        prewitt_result[r+1][c+1] = math.sqrt(prewittx**2 + prewitty**2)

        #find maximum edges
        if (sobel_result[r+1][c+1] > maxsobelEdge):
            maxsobelEdge = sobel_result[r+1][c+1]
            
        if (prewitt_result[r+1][c+1] > maxprewittEdge):
            maxprewittEdge = prewitt_result[r+1][c+1]

#scale all edge values
sobel_result = sobel_result * (255/maxsobelEdge)
prewitt_result = prewitt_result * (255/maxprewittEdge)
cv2.imwrite("test3_sobel.jpg",sobel_result)
cv2.imwrite("test3_prewitt.jpg",prewitt_result)

sobel_thinned = np.zeros([rows,columns])
prewitt_thinned = np.zeros([rows,columns])

for r in range (1,rows-1):
    for c in range (1,columns-1):
        #check if pixel is the local maximum between its vertical neighbours or horizontal neighbours in sobel edges
        if ((sobel_result[r][c] >= sobel_result[r-1][c] and sobel_result[r][c] >= sobel_result[r+1][c]) or (sobel_result[r][c] >= sobel_result[r][c-1] and sobel_result[r][c] >= sobel_result[r][c+1])):
            sobel_thinned[r][c] = sobel_result[r][c]
        #check if pixel is the local maximum between its vertical neighbours or horizontal neighbours in prewitt edges
        if ((prewitt_result[r][c] >= prewitt_result[r-1][c] and prewitt_result[r][c] >= prewitt_result[r+1][c]) or (prewitt_result[r][c] >= prewitt_result[r][c-1] and prewitt_result[r][c] >= prewitt_result[r][c+1])):
            prewitt_thinned[r][c] = prewitt_result[r][c]

cv2.imwrite(os.getcwd() + "/Results/test3_sobel_thinned.jpg",sobel_thinned)
cv2.imwrite(os.getcwd() + "Results/test3_prewitt_thinned.jpg",prewitt_thinned)            
        

            
