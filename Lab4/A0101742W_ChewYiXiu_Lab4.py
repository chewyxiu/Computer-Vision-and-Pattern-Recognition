import os
import cv2
import numpy as np
import math

def rgbtohsv(img,hsv_img):
    row,column,channel = img.shape
    for r in range(0,row):
        for c in range(0,column):

            hue = sat = value = 0.0
            blue,green,red = img[r][c]/255.00        
            cmax = max(blue,green,red)
            cmin = min(blue,green,red) 
            diff = cmax - cmin

            if (diff == 0):
                hue = 0            
            
            elif (cmax == blue):
                hue = 60*(((red-green)/diff) + 4)
                    
            elif (cmax == green):
                hue = 60*(((blue-red)/diff) + 2)

            elif (cmax == red):
                hue = 60*(((green-blue)/diff)%6)

            if (cmax != 0):
                sat = diff/cmax  

            value = cmax
            hsv_img[r][c] = [(hue),(sat),(value)] 
    return

def hsvtorgb(rgb_img,hsv_img):
    row,column,channel = hsv_img.shape
    for r in range(0,row):
        for c in range(0,column):

            C = X = m = red = green = blue = 0

            hue,sat,value = hsv_img[r][c]
            C = value*sat
            X = C*(1-abs(((hue/60)%2)-1))
            m = value - C

            
            if (0 <= hue < 60):
                red = C
                green = X
            
            elif (60 <= hue < 120):
                red = X
                green = C
            
            elif (120 <= hue < 180):
                green = C
                blue = X
            
            elif (180 <= hue < 240):
                green = X
                blue = C
            
            elif (240 <= hue < 300):
                red = X
                blue = C
            elif (300 <= hue < 360):
                red = C
                blue = X

            rgb_img[r][c] = [((blue+m)*255),((green+m)*255),((red+m)*255)]
    return

#read flower image
flower_img = cv2.imread(os.getcwd() + "/Pictures/flower.jpg")
#define shape of array
rows,columns,channels = flower_img.shape
#initialize hsv and rgb arrays for flower img
hsv_flower_img = np.zeros([rows,columns,channels]) 
rgb_flower_img = np.zeros([rows,columns,channels])

#convert flower image to HSV
rgbtohsv(flower_img,hsv_flower_img)
#convert back HSV flower image to RGB
hsvtorgb(rgb_flower_img,hsv_flower_img)

#Display Hue component of HSV image
#Scale hsv image down to [1,0]for imshow
cv2.imshow('Hue Component', hsv_flower_img[:,:,0]/360)
cv2.waitKey(0)
cv2.imwrite(os.getcwd() + "/Results/hue.jpg",hsv_flower_img[:,:,0]*(255.0/360))

#Display Saturation component of HSV image
cv2.imshow('Saturation Component', hsv_flower_img[:,:,1])
cv2.waitKey(0)
#Scale saturation to 255 to write image
cv2.imwrite(os.getcwd() + "/Results/saturation.jpg",hsv_flower_img[:,:,1]*255)


#Display Value component of HSV image
cv2.imshow('Brightness Component', hsv_flower_img[:,:,2])
cv2.waitKey(0)
#Scale value to 255 to write image
cv2.imwrite(os.getcwd() + "/Results/brightness.jpg",hsv_flower_img[:,:,2]*255)

cv2.destroyAllWindows()

#write rgb converted image
cv2.imwrite(os.getcwd() + "/Results/hsv2rgb.jpg",rgb_flower_img)

#read bee image
bee_img = cv2.imread(os.getcwd() + "/Pictures/bee1.png")
#define shape of array
rows,columns,channels = bee_img.shape
#intialize result arrays
hsv_bee_img = np.zeros([rows,columns,channels])
valueComponent = np.zeros([rows,columns])
histeq_img = np.zeros([rows,columns,channels])

#convert bee image to HSV
rgbtohsv(bee_img,hsv_bee_img)

#value component of HSV image scale to 255
valueComponent = hsv_bee_img[:,:,2]*255.0

#set type as int for histogram equalization
valueComponent = valueComponent.astype(int)

hist,bins = np.histogram(valueComponent.flatten(),256,[0,256])
#cummulative sum for histogram value
cdf = hist.cumsum()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
#histogram equalized values
valueComponent = cdf[valueComponent]/255.0

#store value component to value channel in HSV image
hsv_bee_img[:,:,2] = valueComponent
#convert HSV image to RGB with histogram equalized values
hsvtorgb(histeq_img,hsv_bee_img)
#write rgb image after histogram equalization
cv2.imwrite(os.getcwd() + "/Results/histeq.jpg",histeq_img)

