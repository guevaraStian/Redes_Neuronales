# Ejemplo de red neuronal con datos de crimenes por estados de USA DE 1995
# pip install ucimlrepo
# pip install scikit-learn
# pip 25.1.1
# Python 3.13.1

from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Descargamos los datos a una variable dataframe
dataframe = fetch_ucirepo(id=183) 
  
# Extraemos las columnas que vamos a usar en una variable
df = dataframe.data.features 
X = dataframe.data.targets 
Y = df['state']

# metadata 
# print(communities_and_crime.metadata) 

# variable information 
# print(communities_and_crime.variables) 

# Se pasa de texto a números
encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)

# Escalar las características con decimales
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#print(Y_encoded)
#print(X_scaled)

# Se saca en variables aparte una catidad de datos random, relacionados a los que tenemos en dataframe
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_encoded, test_size=0.2, random_state=42)

#print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)

# Creamos el modelo de prediccion con 10 neuronas e iteracion maxima de 500
model = MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu', solver='adam', max_iter=500)

# Entrenamos el modelo de red neuronal con datos X y Y para luego sacar una prediccion en Y_pred
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

#print(model)
#print(Y_pred)

# Obtenemos dato porcentual sobre la exactitud de la red neuronal y la multiplicamos por 100 por el %
exactitud = accuracy_score(Y_test, Y_pred)
exactitud2 = exactitud * 100 

# Se imprime en pantalla el porcentuaje de exactitud
print(f"Exactitud del software : {exactitud2:.2f}  %")

# Mostramos en pantalla un detallado de la clasificacion
print("\nReporte de clasificación:\n", classification_report(Y_test, Y_pred))




