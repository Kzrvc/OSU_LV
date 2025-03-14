import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")
plt.imshow(img)
plt.title("Originalna slika")
plt.show()

bright_img = np.clip(img*1.9, 0, 255).astype(np.uint8)
plt.imshow(bright_img)
plt.title("Posvijetljena slika")
plt.show()

h,w = img.shape[:2]
cropped_img = img[:, w//4:w//2]
plt.imshow(cropped_img)
plt.title("Druga četvrtina slike")
plt.show()

rotated_img = np.rot90(img, k=-1)
plt.imshow(rotated_img)
plt.title("Rotirana slika")
plt.show()

mirrored_img = img[:, ::-1]
plt.imshow(mirrored_img)
plt.title("Zrcalna slika")
plt.show()