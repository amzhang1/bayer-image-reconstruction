import cv2
import numpy as np
import math
from matplotlib import pyplot as plt


# part I

img = cv2.imread('PeppersBayerGray.bmp', 0)

h,w = img.shape

# our final image will be a 3 dimentional image with 3 channels
rgb = np.zeros((h,w,3),np.uint8);


# reconstruction of the green channel IG

IG = np.copy(img) # copy the image into each channel
IR = np.copy(img) # copy the image into each channel
IB = np.copy(img) # copy the image into each channel

for row in range(0,h,4): # loop step is 4 since our mask size is 4.
    for col in range(0,w,4): # loop step is 4 since our mask size is 4.
        # B = (A+C)/2
        IG[row,col+1]=(int(img[row,col])+int(img[row,col+2]))/2
		# D = (C+H)/2
        IG[row,col+3]= (int(img[row,col+2])+int(img[row+1,col+3]))/2
        # M = (I+N)/2
        IG[row+3,col]= (int(img[row+2,col])+int(img[row+3,col+1]))/2
        # E = (A+I) /2 #
        IG[row+1,col]=(int(img[row,col])+int(img[row+2,col]))/2
        # L = (H+P) /2
        IG[row+2,col+3]=(int(img[row+1,col+3])+int(img[row+3,col+3]))/2
        # O = (N+P) /2
        IG[row+3,col+2]=(int(img[row+3,col+1])+int(img[row+3,col+3]))/2
        # G = (F+C+H+K)/4
        IG[row+1,col+2]=(int(img[row+1,col+1])+int(img[row,col+2])+int(img[row+1,col+3])+int(img[row+2,col+2]))/4
        # J = (I+F+K+N)/4
        IG[row+2,col+1]=(int(img[row+2,col])+int(img[row+1,col+1])+int(img[row+2,col+2])+int(img[row+3,col+1]))/4

# reconstruction of the red channel IR

for rows in range(0,h,4):
    for cols in range(0,w,4):
        # C = (B+D)/2
        IR[rows,cols+2]=(int(img[rows,cols+1])+int(img[rows,cols+3]))/2
        # F = (B+J)/2
        IR[rows+1,cols+1]=(int(img[rows,cols+1])+int(img[rows+2,cols+1]))/2
        # H = (D+L)/2
        IR[rows+1,cols+3]=(int(img[rows,cols+3])+int(img[rows+2,cols+3]))/2
        # K = (J+L)/2
        IR[rows+2,cols+2]=(int(img[rows+2,cols+1])+int(img[rows+2,cols+3]))/2
        # G = (B+D+L+J)/4
        IR[rows+1,cols+2]=(int(img[rows,cols+1])+int(img[rows,cols+3])+int(img[rows+2,cols+3])+int(img[rows+2,cols+1]))/4
        # we need to copy the second column and the second last row
        # for the first row and the last row
        IR[rows,cols]= IR[rows,cols+1] # A-->B
        IR[rows+1,cols]= IR[rows+1,cols+1] # E-->F
        IR[rows+2,cols] = IR[rows+2,cols+1] # I-->J
        IR[rows+3,cols+1] = IR[rows+2,cols+1] # N-->J
        IR[rows+3,cols] = IR[rows+2,cols+1] # M-->J
        IR[rows+3,cols+2] = IR[rows+2,cols+2] # O-->K
        IR[rows+3,cols+3]= IR[rows+2,cols+3] # P-->L

# reconstruction of the blue channel IB


for rowz in range(0,h,4):
    for colz in range(0,w,4):
        # F = (E+G)/2
        IB[rowz+1,colz+1] = (int(img[rowz+1,colz])+int(img[rowz+1,colz+2]))/2
        # I = (E+M)/2
        IB[rowz+2,colz] = (int(img[rowz+1,colz])+int(img[rowz+3,colz]))/2
        # J = (E+G+O+M)/4
        IB[rowz+2,colz+1] = (int(img[rowz+1,colz])+int(img[rowz+1,colz+2])+int(img[rowz+3,colz+2])+int(img[rowz+3,colz]))/4
        # K = (G+O)/2
        IB[rowz+2,colz+2] = (int(img[rowz+1,colz+2])+int(img[rowz+3,colz+2]))/2
        # N = (M+O)/2
        IB[rowz+3,colz+1] = (int(img[rowz+3,colz])+int(img[rowz+3,colz+2]))/2
        IB[rowz,colz] = IB[rowz+1,colz]# A-->E
        IB[rowz,colz+1] = IB[rowz+1,colz+1] # B-->F
        IB[rowz,colz+2] = IB[rowz+1,colz+2] # C-->G
        IB[rowz,colz+3] = IB[rowz+1,colz+2] # D-->G
        IB[rowz+1,colz+3] = IB[rowz+1,colz+2] # H-->G
        IB[rowz+2,colz+3] = IB[rowz+2,colz+2] # L-->K
        IB[rowz+3,colz+3] = IB[rowz+3,colz+2] # P-->O

# merge the channels #BGR
rgb[:,:,0]=IR
rgb[:,:,1]=IG
rgb[:,:,2]=IB

cv2.imwrite('rgb.jpg',rgb);

plt.imshow(rgb),plt.title('rgb')
plt.show()

# part II should be written here:
#DR = IR - IG
#DB = IB - IG
#DR = np.float32(DR) # converting channel's type to type float
#DB = np.float32(DB) # converting channel's type to type float

#MR = cv2.medianBlur(DR,3) # applying median filter of size 3x3
#MB = cv2.medianBlur(DB,3) # applying median filter of size 3x3
#MR = np.float32(MR)
#MB = np.float32(MB)

#IRR = MR + IG
#IBB = MB + IG
#IRR = np.uint8(IRR)
#IBB = np.uint8(IBB)
#IRR[IRR > 255.0] = 255 # upper bound is 255, lower bound is 0
#IRR[IRR < 0.0] = 0
#IBB[IBB > 255.0] = 255
#IBB[IBB < 0.0] = 0

#reconstructed = np.zeros((h,w,3),np.uint8);
#reconstructed[:,:,0]=IR
#reconstructed[:,:,1]=IG
#reconstructed[:,:,2]=IBB

#plt.subplot(121),plt.imshow(rgb),plt.title('rgb')
#plt.subplot(122),plt.imshow(reconstructed),plt.title('reconstructed')
#plt.show()
