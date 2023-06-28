import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Import data
sales_df = pd.read_csv('datos_de_ventas.csv')

# Visualization
sns.scatterplot(sales_df['Temperature'], sales_df['Revenue'])

# Creating training set
x_train = sales_df['Temperature']
y_train = sales_df['Revenue']

# Creating model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1, input_shape=[1]))

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error')

# Training
epochs_hist = model.fit(x_train, y_train, epochs=1000)

keys = epochs_hist.history.keys()

# Training graphic model
plt.plot(epochs_hist.history['loss'])
plt.title('Progreso de perdida durante entrenamiento')
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.legend('Training loss')

weights = model.get_weights()
print(weights)

# Prediction
Temp = 30
Revenue = model.predict([Temp])
print(f'La ganancia segun la red neuronal sera de: {Revenue}')

# Prediction Graphic
plt.scatter(x_train, y_train, color='gray')
plt.plot(x_train, model.predict(x_train), color='red')
plt.ylabel('Ganancia [US]')
plt.xlabel('Temperatura [degC]')
plt.title('Ganancia generada vs Temperatura en ambiente')




