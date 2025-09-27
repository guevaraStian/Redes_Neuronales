# Ejemplo de prediccion de red neuronal con datos de tasa de cambio de moneda Colombiana
# Se instalan las librerias necesarias
# pip install pandas numpy matplotlib scikit-learn openpyxl
# pip 25.1.1
# Python 3.13.1
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime

# Con el siguiente codigo se lee el archivo excel de la base de datos
df = pd.read_excel("Tasa_Cambio_Moneda_Colombiana_20250926.xlsx")
Tiempo_Inicial = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # año-mes-día_hora-minuto-segundo
# Se le da nombre a cada columna
df.columns = ['FechaVigencia', 'PromedioValor']

# La columna fecha se guarda en la una variable del df en forma de fecha
df['FechaVigencia'] = pd.to_datetime(df['FechaVigencia'], dayfirst=True)

# Con el siguiente codigo se limpia la columna de la tasa reemplazando los signos
df['PromedioValor'] = (
    df['PromedioValor'].astype(str)
    .str.replace('.', '', regex=False)
    .str.replace(',', '.', regex=False)
    .str.replace('$', '', regex=False)
    .astype(float)
)

# Se organizan las variables que van a entrar a la red neuronal
valores = df['PromedioValor'].values.reshape(-1,1)
scaler = MinMaxScaler()
valores_norm = scaler.fit_transform(valores).flatten()

ventana = 10
X = np.array([valores_norm[i:i+ventana] for i in range(len(valores_norm)-ventana)])
y = np.array([valores_norm[i+ventana] for i in range(len(valores_norm)-ventana)])

# Se entrena modelo MLP sencillo
modelo = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000, random_state=1)
modelo.fit(X, y)

# Con el siguiente codigo se predice la siguiente cifra
y_pred = modelo.predict(X)
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))
y_real_inv = scaler.inverse_transform(y.reshape(-1,1))

# En el siguiente codigo se predice el valor del siguiente dia
ultimos = valores_norm[-ventana:].reshape(1, -1)
pred_sig = modelo.predict(ultimos)
pred_sig_real = scaler.inverse_transform(pred_sig.reshape(-1,1))[0][0]
print(f"\n Predicción para el siguiente día {Tiempo_Inicial}: COP {pred_sig_real:,.2f}")

# En esta parte del codigo se organiza la grafica y sus variables
fechas = df['FechaVigencia'].iloc[ventana:]
plt.figure(figsize=(10,5))
plt.plot(fechas, y_real_inv, label='Valor Real')
plt.plot(fechas, y_pred_inv, label='Valor Predicho', linestyle='--')
plt.title('Predicción USD/COP')
plt.xlabel('Fecha')
plt.ylabel('COP')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Prediccion_TasaDeCambio_MonedaColombiana_{Tiempo_Inicial}.png')
plt.show()

