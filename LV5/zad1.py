'''Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
 ulazne veliˇ cine. Podaci su podijeljeni na skup za uˇ cenje i skup za testiranje modela.'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression 
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

'''a) Prikažite podatke za uˇ cenje u x1−x2 ravnini matplotlib biblioteke pri ˇ cemu podatke obojite
 s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
 marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
 cmap kojima je mogu´ ce definirati boju svake klase.'''

plt.figure()
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Podaci za učenje i testiranje")
plt.legend()
plt.show()

''' b) Izgradite model logistiˇ cke regresije pomo´ cu scikit-learn biblioteke na temelju skupa poda
taka za uˇ cenje.'''

logic_model = LogisticRegression()
logic_model.fit(X_train, y_train)

''' c) Prona¯ dite u atributima izgra¯ denog modela parametre modela. Prikažite granicu odluke
 nauˇcenog modela u ravnini x1 −x2 zajedno s podacima za uˇcenje. Napomena: granica
 odluke u ravnini x1−x2 definirana je kao krivulja: θ0+θ1x1+θ2x2 = 0.'''

theta0 = logic_model.intercept_[0]
theta1 = logic_model.coef_[0][0]
theta2 = logic_model.coef_[0][1]
print(f'Theta 0: {theta0}\nTheta 1: {theta1}\nTheta2: {theta2}')

x1 = np.array([X[:, 0].min(), X[:, 1].max()])
x2 = (-theta0 - (theta1*x1)) / theta2

plt.figure()
plt.plot(x1, x2, color='black', label='Granica odluke')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', marker='o', label='Train')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test')
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Podaci za učenje i testiranje s granicom odluke")
plt.legend()
plt.show()

''' d) Provedite klasifikaciju skupa podataka za testiranje pomo´ cu izgra¯ denog modela logistiˇ cke
 regresije. Izraˇ cunajte i prikažite matricu zabune na testnim podacima. Izraˇ cunate toˇ cnost,
 preciznost i odziv na skupu podataka za testiranje.'''

y_test_p = logic_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_p)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
disp.plot()
plt.show()
print(classification_report(y_test, y_test_p))

'''e) Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznaˇ cite dobro klasificirane
 primjere dok pogrešno klasificirane primjere oznaˇ cite crnom bojom.'''

plt.figure()
plt.scatter(X_test[y_test == y_test_p, 0], X_test[y_test == y_test_p, 1], c='green', label='Dobri')
plt.scatter(X_test[y_test != y_test_p, 0], X_test[y_test != y_test_p, 1], c='black', label='Loši')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Dobro i pogrešno kvalificirani podaci')
plt.legend()
plt.show()




