import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os
th=3
min_p=4
max_p=150

#num=1
i='g'+str(num)+'.jpg'
o='i'+str(num)+'.jpg'
path='/home/aryan/Desktop/Stars/g'
img=cv2.imread(os.path.join(path , i),0)

#img=cv2.imread('g1.jpg',0)
#img=cv2.GaussianBlur(imgx,(5,5),0)
"""
cv2.imshow('Actual Stars',imgx)
cv2.imshow('Filtered Stars',img)
cv2.waitKey(0)
cv2.destroyAllWindows()  """

plt.imshow(img,cmap='gray')
plt.show()
chk=np.zeros(img.shape)
m=img.shape[0]
n=img.shape[1]
def grow(x,y,s):
    if(x<0 or x>=m-1 or y<0 or y>=n-1):
        return;
    if(img[x-1][y]>th and chk[x-1][y]==0 and len(s)<max_p):
        chk[x-1][y]=1
        s.append([x-1,y])
        grow(x-1,y,s)
    if(img[x+1][y]>th and chk[x+1][y]==0 and len(s)<max_p):
        chk[x+1][y]=1
        s.append([x+1,y])
        grow(x+1,y,s)
    if(img[x][y-1]>th and chk[x][y-1]==0 and len(s)<max_p):
        chk[x][y-1]=1
        s.append([x,y-1])
        grow(x,y-1,s)
    if(img[x][y+1]>th and chk[x][y+1]==0 and len(s)<max_p):
        chk[x][y+1]=1
        s.append([x,y+1])
        grow(x,y+1,s)
    return;

s1=[]
def region():
    i=0
    while(i<m):
        j=0
        while(j<n):
            if(img[i,j]>th and chk[i,j]==0):
                chk[i,j]=1
                s=[]
                s.append([i,j])
                grow(i,j,s)
                if(len(s)>=min_p and len(s)!=max_p):
                    s1.append(s)
            j=j+3
        i=i+3

def centroid(s,c):
    x=y=w=0.0
    for i in range(len(s)):
        x+=s[i][0]*img[s[i][0]][s[i][1]]
        y+=s[i][1]*img[s[i][0]][s[i][1]]
        w+=img[s[i][0]][s[i][1]]
    c.append([x/w,y/w])
    return;

start_time = time.time()   
region() 
c=[]   
for j in range(len(s1)):
    centroid(s1[j],c)
 
print("--- %s seconds ---" % (time.time() - start_time)) 
print('Number of stars detected (Threshold value='+str(th)+'): '+str(len(c)))
#print(c)
c.sort()

img1=np.zeros(img.shape,dtype=np.uint8)
cv2.imwrite('Star2.jpg',img1)
img1=cv2.imread('Star2.jpg',0)
for i in range(len(c)):
    m=int(round(c[i][0]))
    n=int(round(c[i][1]))
    cv2.circle(img1,(n,m),2,(255,255,255))
"""cv2.imshow('Indentified Stars',img1)
cv2.imshow('Actual Stars',img)
cv2.imwrite('Act.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()   """

path='/home/aryan/Desktop/Stars/i'
cv2.imwrite(os.path.join(path , o),img1) 