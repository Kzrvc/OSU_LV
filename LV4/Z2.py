#Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoričku varijable „Fuel Type“ kao ulaznu veličinu. Pri tome koristite 1-od-K kodiranje kategoričkih veličina. 
#Radi jednostavnosti nemojte skalirati ulazne veličine. Komentirajte dobivene rezultate.
#Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? O kojem se modelu vozila radi?

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#dio prvog zadatka koji nam je potreban

df = pd.read_csv("data_C02_emission.csv")

features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']
target = 'CO2 Emissions (g/km)'

#drugi zadatak
new_features = features + ['Fuel Type']

X = df[new_features]
y = df[target]

X_encoded = pd.get_dummies(X, columns=['Fuel Type'])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
max_error = max(abs(y_test - y_pred))
max_error_index = abs(y_test - y_pred).idxmax()

print("Stvarna vrijednost:", y_test.loc[max_error_index])
print("Predviđena vrijednost:", y_pred[list(y_test.index).index(max_error_index)])
print(f"MSE = {mse:.2f}, MAE = {mae:.2f}, R² = {r2:.4f}")
print(f"Max error = {max_error:.2f} g/km")
print("Model vozila s najvećom pogreškom:", df.loc[max_error_index])