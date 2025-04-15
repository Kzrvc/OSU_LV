import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()



colors=np.unique(img_array, axis=0)
num_colors=colors.shape[0]
print(f"broj različitih boja je: {num_colors}")

K=5
kmeans=KMeans(n_clusters=K,random_state=0)
kmeans.fit(img_array)

centrals=kmeans.cluster_centers_
labels=kmeans.predict(img_array)
img_array_aprox=centrals[labels]
img_array_aprox=np.reshape(img_array_aprox,(w,h,d))

plt.subplot(1,2,1)
plt.title("orginal")
plt.imshow(img)
plt.axis("off")

plt.subplot(1,2,2)
plt.title("aprox")
plt.imshow(img_array_aprox)
plt.axis("off")
plt.show()

image_path=["imgs\\test_2.jpg","imgs\\test_3.jpg","imgs\\test_4.jpg","imgs\\test_5.jpg","imgs\\test_6.jpg"]

for path in image_path:
    img = Image.imread(path)
    if(path!="imgs\\test_4.jpg"):
        img = img.astype(np.float64) / 255
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))
    img_array_aprox = img_array.copy()
    kmeans=KMeans(n_clusters=K,random_state=0)
    kmeans.fit(img_array)

    centrals=kmeans.cluster_centers_
    labels=kmeans.predict(img_array)
    img_array_aprox=centrals[labels]
    img_array_aprox=np.reshape(img_array_aprox,(w,h,d))
    plt.subplot(1,2,1)
    plt.title("orginal")
    plt.imshow(img)
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.title("aprox")
    plt.imshow(img_array_aprox)
    plt.axis("off")
    plt.show()
    
    
    wcss=[]
    for i in range(1,11):
        kmeans=KMeans(n_clusters=i, random_state=52)
        kmeans.fit(img_array)
        wcss.append(kmeans.inertia_) 

    plt.figure()
    plt.plot(range(1,11),wcss)
    plt.title('lakat')
    plt.xlabel('K')
    plt.ylabel('wcss')
    plt.show()



K=3
kmeans=KMeans(n_clusters=K,random_state=0)
kmeans.fit(img_array)
labels=kmeans.predict(img_array)

for i in range(K):
    mask=(labels==i).astype(np.uint8)
    mask_img=mask.reshape(w,h)

    plt.figure()
    plt.imshow(mask_img,cmap='gray')
    plt.title("binarna")
    plt.axis("off")
    plt.show()
    
plt.figure()
plt.imshow(mask_img,cmap='gray')
plt.title("binarna")
plt.axis("off")
plt.show()



















'''
zaključak
Na prvoj slici imamo 97924 različitih boja


Na slici sa 5 boja vidimo da ima manje detalja ali je i dalje dosta slična orginalu

Možemo uočiti lakat za svaku sliku. Nijedan nije prezaobljen pa možemo.

'''