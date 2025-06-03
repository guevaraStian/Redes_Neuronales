# Ejemplo de red neuronal sencilla
# pip install pandas seaborn
# pip 25.1.1
# Python 3.13.1

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv('BasesDeDatos/iris.csv')
#df = sns.load_dataset('BasesDeDatos/iris.csv')
#df = sns.load_dataset('iris')
#print(df)

new_names = {'5.1': 'Longitud_Sepalo_Cm', '3.5': 'Ancho_Sepalo_Cm', '1.4': 'Longitud_Petalo_Cm', '0.2':'Ancho_Petalo_Cm', 'Iris-setosa':'Especie'}                        

# Renombrar las columnas
df = df.rename(columns=new_names)

# Elegir las columnas a graficar
x = df['Ancho_Petalo_Cm']
y = df['Especie']

# Crear el gráfico
# plt.figure(figsize=(8, 5))
# plt.plot(x, y, marker='o', linestyle='-', color='teal')
# plt.bar(x, y)
# plt.hist(y)
# plt.scatter(x, y)
# plt.pie(x, y)
# plt.boxplot(df)

# plt.title('Flor y ancho del petalo')
# plt.xlabel('Ancho_Petalo_Cm')
# plt.ylabel('Especie')
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# Se divide la informacion entre características (X) y etiquetas (y)
X = df.drop('Especie', axis=1)
y = df['Especie']

# Se pasa de texto a números
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Escalar las características (opcional pero recomendado para redes neuronales)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Se extraen las variables de prueba e iteracion con la tabla
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)


# Se crea y se entrena la red neuronal
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500)
model.fit(X_train, y_train)

# Se predice la proxima respuesta
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Se imprime en pantalla la respuesta de la red neuronal
print(f"Precisión: {accuracy:.2f}")
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred, target_names=encoder.classes_))






