import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("Lv3/data_C02_emission.csv")

#a)Pomoću histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
data['CO2 Emissions (g/km)'].plot(kind = 'hist', color = 'skyblue', edgecolor = 'gray')
plt.title("Diagram CO2 Emissions")
plt.show()

#b)Pomoću dijagrama raspršenja prikažite odnos između gradske potrošnje goriva i emisije C02 plinova.
#Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose između veličina, obojite točkice na dijagramu raspršenja s obzirom na tip goriva.
data['Fuel Type'] = data['Fuel Type'].astype('category')
data.plot.scatter(x = 'Fuel Consumption City (L/100km)',
                  y = 'CO2 Emissions (g/km)',
                  c = 'Fuel Type', cmap = 'cool', s = 20)
plt.title("Diagram Fuel Consumption - CO2 Emission")
plt.show()

#c)Pomoću kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva. 
#Primjećujete li grubu mjernu pogrešku u podacima?
data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.show()

#d)Pomoću stupčastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby.
fuel_counts = data.groupby('Fuel Type').size()
fuel_counts.plot(kind='bar', color="lightblue", edgecolor="purple")
plt.title("Diagram Vehicles - Fuel Type")
plt.show()

#e)Pomoću stupčastog grafa prikažite na istoj slici prosječnu C02 emisiju vozila s obzirom na broj cilindara.
avg_co2 = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
avg_co2.plot(kind = 'bar', color='lightpink',edgecolor='purple')
plt.title("Diagram CO2 Emission - Cylinders")
plt.show()