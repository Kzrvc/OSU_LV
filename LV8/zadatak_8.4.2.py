# Zadatak 8.4.2 Napišite skriptu koja  ce ucitati izgradenu mrežu iz zadatka 1 i MNIST skup
# podataka. Pomocu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
# skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvidenu
# mrežom.

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Učitaj testne podatke i pripremi ih
(_, _), (x_test, y_test) = mnist.load_data()

# Skaliraj slike na [0, 1] i dodaj kanal dimenziju (28, 28, 1)
x_test = x_test.astype("float32") / 255
x_test = np.expand_dims(x_test, -1)

# Učitaj prethodno trenirani model
model = load_model("model_mnist_fcn.keras")

# Predikcija modela
y_pred_probs = model.predict(x_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# Pronađi pogrešne klasifikacije
wrong_indexes = np.where(y_pred_classes != y_test)[0]

# Prikaži prvih 10 pogrešnih klasifikacija
for i in range(10):
    idx = wrong_indexes[i]
    plt.imshow(x_test[idx].reshape(28, 28), cmap="gray")
    plt.title(f"Stvarno: {y_test[idx]}, Predikcija: {y_pred_classes[idx]}")
    plt.axis('off')
    plt.show()