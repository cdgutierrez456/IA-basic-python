import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Importando Datos
temperature_df = pd.read_csv("celsius_a_fahrenheit.csv")

# Visualizacion
sns.scatterplot(temperature_df['Celsius'], temperature_df['Fahrenheit'])

# Cargando set de datos
x_train = temperature_df['Celsius']
y_train = temperature_df['Fahrenheit']

# Creando el Modelo
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

#model.summary()

# Compilado
model.compile(optimizer=tf.keras.optimizers.Adam(1), loss='mean_squared_error')

# Entrenando el modelo
epochs_hist = model.fit(x_train, y_train, epochs = 100)

# Evaluando el modelo
epochs_hist.history.keys()

# Grafico
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de perdida durante entrenamiento del modelo')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend('Training Loss')

model.get_weights()

# Predicciones
Temp_C = 0
Temp_F = model.predict([Temp_C])
print(Temp_F)

# Formula real
Temp_F = 9/5 * Temp_C + 32
print(Temp_F)


