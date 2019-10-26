import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
threshold=3     #Threshold defined in terms of pixel values
min_pixel=4     #Minimum number of pixels that star must have to be identified
max_pixel=150       #Maximum number of pixels that star must have to be a valid star

num=1
i='g'+str(num)+'.jpg'
o='i'+str(num)+'.jpg'
path='/home/aryan/Desktop'      #Input image path defined
img=cv2.imread(os.path.join(path , i),0)   #read the input image

plt.imshow(img,cmap='gray')     #Show the input image
plt.show()
chk=np.zeros(img.shape)     #Checking matrix (To mark visited places in image)
m=img.shape[0]
n=img.shape[1]
def grow(x,y,s):        #Region growth algorithm: Inp= seed location(x,y) and an empty 1D array, Out= 1D array containing pixel locations
    if(x<0 or x>=m-1 or y<0 or y>=n-1):
        return;
    if(img[x-1][y]>threshold and chk[x-1][y]==0 and len(s)<max_pixel):
        chk[x-1][y]=1
        s.append([x-1,y])
        grow(x-1,y,s)
    if(img[x+1][y]>threshold and chk[x+1][y]==0 and len(s)<max_pixel):
        chk[x+1][y]=1
        s.append([x+1,y])
        grow(x+1,y,s)
    if(img[x][y-1]>threshold and chk[x][y-1]==0 and len(s)<max_pixel):
        chk[x][y-1]=1
        s.append([x,y-1])
        grow(x,y-1,s)
    if(img[x][y+1]>threshold and chk[x][y+1]==0 and len(s)<max_pixel):
        chk[x][y+1]=1
        s.append([x,y+1])
        grow(x,y+1,s)
    return;

stars=[]  #empty 2D array
def region():       #Iteratively calls the region growth algorithm for each identified star
    i=0
    while(i<m):
        j=0
        while(j<n):
            if(img[i,j]>threshold and chk[i,j]==0):
                chk[i,j]=1
                s=[]
                s.append([i,j])
                grow(i,j,s)
                if(len(s)>=min_pixel and len(s)!=max_pixel):
                    stars.append(s)
            j=j+3
        i=i+3

def centroid(s,c):      #Centroiding algorithm (Weighted means)
    x=y=w=0.0
    for i in range(len(s)):
        x+=s[i][0]*img[s[i][0]][s[i][1]]
        y+=s[i][1]*img[s[i][0]][s[i][1]]
        w+=img[s[i][0]][s[i][1]]
    c.append([x/w,y/w])
    return;

start_time = time.time()   
region() 
centroids=[]   
for j in range(len(stars)):
    centroid(stars[j],centroids)
 
print("--- %s seconds ---" % (time.time() - start_time))        #Display execution time
print('Number of stars detected (Threshold value='+str(threshold)+'): '+str(len(centroids)))
centroids.sort()

img1=np.zeros(img.shape,dtype=np.uint8)
cv2.imwrite('Star2.jpg',img1)
img1=cv2.imread('Star2.jpg',0)
for i in range(len(centroids)):
    m=int(round(centroids[i][0]))
    n=int(round(centroids[i][1]))
    cv2.circle(img1,(n,m),2,(255,255,255))

path='/home/aryan/Desktop'      
cv2.imwrite(os.path.join(path , o),img1)        #Save the identified image