import numpy as np
import matplotlib.pyplot as plt
image = np.zeros((480,640))
import cv2

def f(x, y, H, sigma, x_0, y_0):
    return H * (np.exp(-((x - x_0) ** 2 + (y - y_0) ** 2) / (2 * sigma ** 2)))/ (2 * np.pi * sigma ** 2)

def H(M, k_1 = 10 ** 5, k_2 = 1, k_3 = 1):
    return k_1 * np.exp(- k_2 * M + k_3)

f_noise_max = 30
f_noise_min = 0
def r(f_n_max = f_noise_max, f_n_min = f_noise_min):
    return f_n_min + (f_n_max - f_n_min) * np.random.random()

n = 25  #Number of stars
x = image.shape[1] * np.random.random(n)
y = image.shape[0] * np.random.random(n)
b = 6 * np.random.random(n)

cen=[]
#print(image.shape)
#print(480-y) #actually x-coordinate
#print(x)  #actually y-coordinate
for i in range(n):
    cen.append([480-y[i],x[i]])   
print("\n")
cen.sort()
print(cen)
plt.plot(x,y, 'o')
plt.xlim(0,640)
plt.ylim(0,480);
#plt.axis('off');
#plt.savefig('coordinates.jpg', quality = 95, format = 'jpg')

image2 = np.zeros(image.shape,dtype=np.uint8)
sigma = 1.5
for i in np.arange(n):
     m1=int(round(475-y[i]))
     n1=int(round(x[i]-5))
    #for j in np.arange(image2.shape[0]):
     for j in range(20):
        #for k in np.arange(image2.shape[1]):
         for k in range(20):
            image2[m1+j][n1+k] = image2[m1+j][n1+k] + f(m1+j, n1+k, H(b[i]), sigma, 480-y[i], x[i])

cv2.imwrite('gen1.jpg',image2)
"""
image = image2.copy()
for j in np.arange(image.shape[0]):
    for k in np.arange(image.shape[1]):
        image[j][k] = image[j][k] + r()

cv2.imwrite('gen2.jpg',image)"""