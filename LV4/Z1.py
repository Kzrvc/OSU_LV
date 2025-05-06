import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv("data_C02_emission.csv")

#a)Odaberite željene numeričke veličine specificiranjem liste s nazivima stupaca. Podijelite podatke na skup za učenje i skup za testiranje u omjeru 80%-20%.
features = ['Engine Size (L)', 'Cylinders', 'Fuel Consumption City (L/100km)', 'Fuel Consumption Hwy (L/100km)', 'Fuel Consumption Comb (L/100km)']
target = 'CO2 Emissions (g/km)'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#b)Pomoću matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova o jednoj numeričkoj veličini. 
#Pri tome podatke koji pripadaju skupu za učenje označite plavom bojom, a podatke koji pripadaju skupu za testiranje označite crvenom bojom.
plt.scatter(X_train['Engine Size (L)'], y_train, color='blue', label='Train')
plt.scatter(X_test['Engine Size (L)'], y_test, color='red', label='Test')
plt.xlabel('Engine Size (L)')
plt.ylabel('CO2 Emissions (g/km)')
plt.title('CO2 Emissions vs Engine Size')
plt.legend()
plt.show()

#c)Izvršite standardizaciju ulaznih veličina skupa za učenje. Prikažite histogram vrijednosti jedne ulazne veličine prije i nakon skaliranja. 
#Na temelju dobivenih parametara skaliranja transformirajte ulazne veličine skupa podataka za testiranje.
scaler = MinMaxScaler()

plt.hist(X_train['Engine Size (L)'], bins=20, color='gray')
plt.title('Before Scaling: Engine Size')
plt.show()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

plt.hist(X_train_scaled['Engine Size (L)'], bins=20, color='pink')
plt.title('After Scaling: Engine Size')
plt.show()

#d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i povežite ih s izrazom 4.6.
model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("Intercept (θ₀):", model.intercept_)
print("Koeficijenti (θ):", model.coef_)

#e)Izvršite procjenu izlazne veličine na temelju ulaznih veličina skupa za testiranje. 
#Prikažite pomoću dijagrama raspršenja odnos između stvarnih vrijednosti izlazne veličine i procjene dobivene modelom.
y_pred = model.predict(X_test_scaled)
plt.scatter(y_test, y_pred, color='purple')
plt.xlabel('Real value')
plt.ylabel('Predicted value')
plt.title('Real vs predicted value (CO2)')
plt.show()

#f)Izvršite vrednovanje modela na način da izračunate vrijednosti regresijskih metrika na skupu podataka za testiranje
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('MSE: ', mse)
print('MAE: ', mae)
print("R²: ", r2)

#g)Što se dogada s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj ulaznih veličina?
for i, feat in enumerate(features, 1):
    X = df[feat] 
    if isinstance(X, pd.Series): 
        X = X.to_frame()
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
bartol je konjina
    print(f"Model {i} ({', '.join(feat)}): MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

