import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

#return a grayscale image 
img = cv2.imread('/Pictures/pic4.jpg',cv2.CV_LOAD_IMAGE_GRAYSCALE)
#return values of histogram for img
hist,bins = np.histogram(img.flatten(),256,[0,256])
#cummulative sum for histogram value
cdf = hist.cumsum()
cdfFig = plt.figure(1)
histogramFig = plt.figure(2)
#plot before histogram equalization cdf
plt.figure(1)
beforeCdf = cdfFig.add_subplot(4,1,1)
beforeCDf = plt.plot(cdf, color = 'b', label = 'cdf before')
beforeCDf = plt.xlim([0,256])
beforeCDf  = plt.legend(loc = 'upper left')

#plot before histogram equalization histogram
plt.figure(2)
beforeHistogram = histogramFig.add_subplot(4,1,2)
beforeHistogram = plt.hist(img.flatten(),256,[0,256], color = 'r', label = 'histogram before')
beforeHistogram = plt.xlim([0,256])
beforeHistogram = plt.legend(loc = 'upper left')

cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')
img1 = cdf[img]

#return values of histogram after histogram equalization
hist,bins = np.histogram(img1.flatten(),256,[0,256])

#plot after histogram equalization cdf
plt.figure(1)
cdf = hist.cumsum()
afterCdf = cdfFig.add_subplot(4,1,3)
afterCdf = plt.plot(cdf, color = 'b', label = 'cdf after')
afterCdf = plt.xlim([0,256])
afterCdf  = plt.legend(loc = 'upper left')

#plot after histogram equalization histogram
plt.figure(2)
afterHistogram = histogramFig.add_subplot(4,1,4)
afterHistogram = plt.hist(img1.flatten(),256,[0,256], color = 'r',label = 'histogram after')
afterHistogram =plt.xlim([0,256])
afterHistogram = plt.legend(loc = 'upper left')
plt.show()

#left image is before, right image is after histogram equalisation
result = np.hstack((img,img1))
cv2.imwrite("pic4BeforeAfter.jpg",result)
