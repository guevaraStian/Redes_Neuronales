# Ejemplo de red neuronal sencilla
# pip install scikit-learn
# pip 25.1.1
# Python 3.13.1

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar unos datos de Iris
iris = load_iris()
X = iris.data  # Características
y = iris.target  # Etiquetas

# Se extraen datos y se organizan variables de prueba y los de entrenamiento
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Se hace el llamado a el arbol de clasificacion
model = DecisionTreeClassifier()

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train)

# Hacer predicciones con los datos de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print("Precisión del modelo: ", accuracy)

# Probar con un dato nuevo
nuevo_dato = [[0.7, 3.8, 4.6, 2.1]]
prediccion = model.predict(nuevo_dato)
print("Predicción para el nuevo dato: ", iris.target_names[prediccion[0]])



