# Zadatak 8.4.3 Napišite skriptu koja ce ucitati izgradenu mrežu iz zadatka 1. Nadalje, skripta
# treba ucitati sliku test.png sa diska. Dodajte u skriptu kod koji  ́ce prilagoditi sliku za mrežu,
# klasificirati sliku pomo ́cu izgradene mreže te ispisati rezultat u terminal. Promijenite sliku
# pomo ́cu nekog grafickog alata (npr. pomo ́cu Windows Paint-a nacrtajte broj 2) i ponovo pokrenite
# skriptu. Komentirajte dobivene rezultate za razliˇcite napisane znamenke.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Učitaj spremljeni model
model = load_model("model_mnist_fcn.keras")

# Učitaj sliku s diska (pretpostavlja se da se zove 'test.png')
img_path = "test.png"

# Otvori sliku, pretvori u grayscale i promijeni dimenzije na 28x28
img = Image.open(img_path).convert("L").resize((28, 28))

# Prikaz slike u terminalu (opcionalno)
plt.imshow(img, cmap="gray")
plt.title("Učitana slika")
plt.axis("off")
plt.show()

# Pretvori sliku u numpy array i skaliraj na [0, 1]
img_array = np.array(img).astype("float32") / 255.0


# Dodaj kanalsku dimenziju i batch dimenziju → (1, 28, 28, 1)
img_array = np.expand_dims(img_array, axis=-1)
img_array = np.expand_dims(img_array, axis=0)

# Klasifikacija slike
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Ispis rezultata
print(f"Model predviđa da je broj na slici: {predicted_class}")