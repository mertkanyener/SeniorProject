import cv2
import numpy as np

img1 = cv2.imread('img1.bmp')
img2 = cv2.imread('img2.bmp')
img3 = cv2.imread('img3.bmp')
img4 = cv2.imread('img4.bmp')

print("Before resize:")
print("img1 = ", img1.shape)
print("img2 = ", img2.shape)
print("img3 = ", img3.shape)
print("img4 = ", img4.shape)



images = [img1, img2, img3, img4]

biggest = 0
index = 0
for i in range(len(images)):
    hypotenuse = np.sqrt(images[i].shape[0]**2 + images[i].shape[1]**2)
    if hypotenuse > biggest:
        index = i
        biggest = hypotenuse
for j in range(len(images)):
    images[j] = cv2.resize(images[j], (images[index].shape[1], images[index].shape[0]))


print("After resize:")
print("img1 = ", images[0].shape)
print("img2 = ", images[1].shape)
print("img3 = ", images[2].shape)
print("img4 = ", images[3].shape)

cv2.imshow('img1', images[0])
cv2.imshow('img2', images[1])
cv2.imshow('img3', images[2])
cv2.waitKey(0)
cv2.destroyAllWindows()