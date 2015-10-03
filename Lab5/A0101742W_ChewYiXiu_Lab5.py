import os
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

def edgeStrength(img):
    rows,columns = img.shape
    #horizontal edges
    gx = np.zeros([rows,columns])
    #vertical edges
    gy = np.zeros([rows,columns])

    I_xx = np.zeros([rows,columns])
    I_xy = np.zeros([rows,columns])
    I_yy = np.zeros([rows,columns])

    #define sobel kernels
    sobel_vertical_kernel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
    sobel_horizontal_kernel = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])
    for r in range(0,rows-2):
        for c in range(0,columns-2):
            #convolve image with sobel kernels
            gy[r+1][c+1] = (img[r][c]*sobel_vertical_kernel[0][0]) + (img[r][c+1]*sobel_vertical_kernel[0][1]) + (img[r][c+2]*sobel_vertical_kernel[0][2]) + (img[r+1][c]*sobel_vertical_kernel[1][0]) + (img[r+1][c+1]*sobel_vertical_kernel[1][1]) + (img[r+1][c+2]*sobel_vertical_kernel[1][2]) + (img[r+2][c]*sobel_vertical_kernel[2][0]) + (img[r+2][c+1]*sobel_vertical_kernel[2][1]) + (img[r+2][c+2]*sobel_vertical_kernel[2][2])
            gx[r+1][c+1] = (img[r][c]*sobel_horizontal_kernel[0][0]) + (img[r][c+1]*sobel_horizontal_kernel[0][1]) + (img[r][c+2]*sobel_horizontal_kernel[0][2]) + (img[r+1][c]*sobel_horizontal_kernel[1][0]) + (img[r+1][c+1]*sobel_horizontal_kernel[1][1]) + (img[r+1][c+2]*sobel_horizontal_kernel[1][2]) + (img[r+2][c]*sobel_horizontal_kernel[2][0]) + (img[r+2][c+1]*sobel_horizontal_kernel[2][1])+ (img[r+2][c+2]*sobel_horizontal_kernel[2][2])
            
    I_xx = gx * gx
    I_xy = gx * gy
    I_yy = gy * gy        

    return I_xx,I_xy,I_yy

def gaussKernels(size,sigma=1):
    ## returns a 2d gaussian kernel
    if size<3:
        size = 3
    m = size/2
    x, y = np.mgrid[-m:m+1, -m:m+1]
    kernel = np.exp(-(x*x + y*y)/(2*sigma*sigma))
    kernel_sum = kernel.sum()
    if not sum==0:
        kernel = kernel/kernel_sum
    return kernel

def convolveGaussianKernel(I_xx,I_xy,I_yy,gaussian_kernel):
    rows,columns = I_xx.shape
    W_xx = np.zeros([rows,columns])
    W_xy = np.zeros([rows,columns])
    W_yy = np.zeros([rows,columns])

    for r in range(0,rows-2):
        for c in range(0,columns-2):
            #convolve with gaussian kernel
            W_xx[r+1][c+1] = (I_xx[r][c]*gaussian_kernel[0][0]) + (I_xx[r][c+1]*gaussian_kernel[0][1]) + (I_xx[r][c+2]*gaussian_kernel[0][2]) + (I_xx[r+1][c]*gaussian_kernel[1][0]) + (I_xx[r+1][c+1]*gaussian_kernel[1][1]) + (I_xx[r+1][c+2]*gaussian_kernel[1][2]) + (I_xx[r+2][c]*gaussian_kernel[2][0]) + (I_xx[r+2][c+1]*gaussian_kernel[2][1]) + (I_xx[r+2][c+2]*gaussian_kernel[2][2])
            W_xy[r+1][c+1] = (I_xy[r][c]*gaussian_kernel[0][0]) + (I_xy[r][c+1]*gaussian_kernel[0][1]) + (I_xy[r][c+2]*gaussian_kernel[0][2]) + (I_xy[r+1][c]*gaussian_kernel[1][0]) + (I_xy[r+1][c+1]*gaussian_kernel[1][1]) + (I_xy[r+1][c+2]*gaussian_kernel[1][2]) + (I_xy[r+2][c]*gaussian_kernel[2][0]) + (I_xy[r+2][c+1]*gaussian_kernel[2][1]) + (I_xy[r+2][c+2]*gaussian_kernel[2][2])
            W_yy[r+1][c+1] = (I_yy[r][c]*gaussian_kernel[0][0]) + (I_yy[r][c+1]*gaussian_kernel[0][1]) + (I_yy[r][c+2]*gaussian_kernel[0][2]) + (I_yy[r+1][c]*gaussian_kernel[1][0]) + (I_yy[r+1][c+1]*gaussian_kernel[1][1]) + (I_yy[r+1][c+2]*gaussian_kernel[1][2]) + (I_yy[r+2][c]*gaussian_kernel[2][0]) + (I_yy[r+2][c+1]*gaussian_kernel[2][1]) + (I_yy[r+2][c+2]*gaussian_kernel[2][2])
            
    return W_xx,W_xy,W_yy

def findResponse(W_xx,W_xy,W_yy):
    rows,columns = W_xx.shape
    detW = np.zeros([rows,columns])
    traceW = np.zeros([rows,columns])
    response = np.zeros([rows,columns])

    for r in range(0,rows-2):
        for c in range(0,columns-2):
            detW[r][c] = (W_xx[r][c] * W_yy[r][c]) - (W_xy[r][c] * W_xy[r][c])
            traceW[r][c] = (W_xx[r][c] + W_yy[r][c])
            response[r][c] = detW[r][c] - 0.06 * (traceW[r][c] * traceW[r][c])

    maxResponse = np.amax(response)

    #keep only those of at least 10% of the maximum response
    for r in range(0,rows):
        for c in range(0,columns):
            if response[r][c] < 0.1*maxResponse:
                response[r][c] = 0    
    return response

def plotImage(img,response):
    rows,columns,channels = img.shape
    for r in range(0,rows):
        for c in range(0,columns):
            if response[r][c] != 0:
                #set harris corner to pink
                img[r][c] = 255,20,147 
    plt.figure()
    plt.imshow(img)        
    plt.show()
    return img

def BGR2RGB(img):
    rows,columns,channels = img.shape
    rgb_img = np.zeros([rows,columns,channels])
    rgb_img[:,:,0] = img[:,:,2] 
    rgb_img[:,:,1] = img[:,:,1]
    rgb_img[:,:,2] = img[:,:,0]
    rgb_img = rgb_img.astype(np.uint8)
    return rgb_img

def RGB2BGR(img):
    rows,columns,channels = img.shape
    bgr_img = np.zeros([rows,columns,channels])
    bgr_img[:,:,0] = img[:,:,2] 
    bgr_img[:,:,1] = img[:,:,1]
    bgr_img[:,:,2] = img[:,:,0]
    bgr_img = bgr_img.astype(np.uint8)
    return bgr_img

    
#define gaussian kernel
gaussian_kernel = gaussKernels(3,1)

building1_img = cv2.imread(os.getcwd() + "/Pictures/building1.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
I_xx,I_xy,I_yy = edgeStrength(building1_img)
W_xx,W_xy,W_yy = convolveGaussianKernel(I_xx,I_xy,I_yy,gaussian_kernel)
building1_response = findResponse(W_xx,W_xy,W_yy)
#read BGR image
building1_bgr = cv2.imread(os.getcwd() + "/Pictures/building1.png")
building1_rgb = BGR2RGB(building1_bgr)
#plot image with RGB image
building1_result = plotImage(building1_rgb,building1_response)
building1_bgr = RGB2BGR(building1_result)
#write image with BGR image
cv2.imwrite(os.getcwd() + "/Results/building1_harris_corner.png",building1_bgr)

building2_img = cv2.imread(os.getcwd() + "/Pictures/building2.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
I_xx,I_xy,I_yy = edgeStrength(building2_img)
W_xx,W_xy,W_yy = convolveGaussianKernel(I_xx,I_xy,I_yy,gaussian_kernel)
building2_response = findResponse(W_xx,W_xy,W_yy)
#read BGR image
building2_bgr = cv2.imread(os.getcwd() + "/Pictures/building2.png")
building2_rgb = BGR2RGB(building2_bgr)
#plot image with RGB image
building2_result = plotImage(building2_rgb,building2_response)
building2_bgr = RGB2BGR(building2_result)
#write image with BGR image
cv2.imwrite(os.getcwd() + "/Results/building2_harris_corner.png",building2_bgr)

checker_img = cv2.imread(os.getcwd() + "/Pictures/checker.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
I_xx,I_xy,I_yy = edgeStrength(checker_img)
W_xx,W_xy,W_yy = convolveGaussianKernel(I_xx,I_xy,I_yy,gaussian_kernel)
checker_response = findResponse(W_xx,W_xy,W_yy)
#read BGR image
checker_bgr = cv2.imread(os.getcwd() + "/Pictures/checker.jpg")
checker_rgb = BGR2RGB(checker_bgr)
#plot image with RGB image
checker_result = plotImage(checker_rgb,checker_response)
checker_bgr = RGB2BGR(checker_result)
#write image with BGR image
cv2.imwrite(os.getcwd() + "/Results/checker_harris_corner.jpg",checker_bgr)

flower_img = cv2.imread(os.getcwd() + "/Pictures/flower.jpg",cv2.CV_LOAD_IMAGE_GRAYSCALE)
I_xx,I_xy,I_yy = edgeStrength(flower_img)
W_xx,W_xy,W_yy = convolveGaussianKernel(I_xx,I_xy,I_yy,gaussian_kernel)
flower_response = findResponse(W_xx,W_xy,W_yy)
#read BGR image
flower_bgr = cv2.imread(os.getcwd() + "/Pictures/flower.jpg")
flower_rgb = BGR2RGB(flower_bgr)
#plot image with RGB image
flower_result = plotImage(flower_rgb,flower_response)
flower_bgr = RGB2BGR(flower_result)
#write image with BGR image
cv2.imwrite(os.getcwd() + "/Results/flower_harris_corner.jpg",flower_bgr)
