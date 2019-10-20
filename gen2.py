#!/usr/bin/env python
# coding: utf-8

# # Image Generation

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
from random import randint

# In[2]:


image = np.zeros((480,640))
sigma = 1.5

# In[3]:


#plt.imshow(image, cmap='gray', vmin=0, vmax=255, origin = "lower");


# In[4]:

z1=np.sqrt((2 * np.pi * sigma ** 2))
def f(x, y, H, sigma, x_0, y_0):
    return H * (np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / (2 *sigma ** 2)) / z1)


# In[5]:


def H(M, k_1 = 10 ** 5, k_2 = 1, k_3 = 1):
    return k_1 * np.exp(- k_2 * M + k_3)


# In[6]:


f_noise_max = 30
f_noise_min = 0
def r(f_n_max = f_noise_max, f_n_min = f_noise_min):
    return f_n_min + (f_n_max - f_n_min) * np.random.random()


# In[7]:


# Selecting the stars
n = 25
w=50
x = 10 + (image.shape[1] - w-2) * np.random.random(n)
y = 10 + (image.shape[0] - w-2) * np.random.random(n)
b = 6 * np.random.random(n)

# In[8]:
cen=[]
#print(image.shape)
#print(480-y) #actually x-coordinate
#print(x)  #actually y-coordinate
for i in range(n):
    cen.append([480-y[i],x[i]])   
print("\n")
cen.sort()


# In[10]:


plt.plot(x,y, 'o')
plt.xlim(0,640)
plt.ylim(0,480);
#plt.axis('off');
#plt.savefig('coordinates.jpg', quality = 95, format = 'jpg')

# In[11]:


image2 = np.zeros(image.shape)
for i in np.arange(n):
    m1=int(round(480-y[i]-w/2))
    n1=int(round(x[i]-w/2))
    amp=randint(1,n/5)
    #for j in np.arange(image2.shape[0]):
    for j in range(w):
        #for k in np.arange(image2.shape[1]):
         for k in range(w):
            z=f(m1+j, n1+k, (255*5/n)*amp*z1, sigma, 480-y[i], x[i])
            #print(z/H(b[i]))
            image2[m1+j][n1+k] = image2[m1+j][n1+k] + z

image = image2.copy()
for j in np.arange(image.shape[0]):
    for k in np.arange(image.shape[1]):
        image[j][k] = image[j][k] + r()


# In[12]:


cv2.imwrite('g1.jpg', image2)


# In[ ]:



