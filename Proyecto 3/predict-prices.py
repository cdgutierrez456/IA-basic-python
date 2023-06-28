import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
## MinMaxScaler para escalado
from sklearn.preprocessing import MinMaxScaler
## train_test_split para entrenamiento
from sklearn.model_selection import train_test_split

# Importing data
house_df = pd.read_csv('precios_hogares.csv')

# Visualization
sns.scatterplot(x = 'sqft_living', y = 'price', data = house_df)

# Correlation
f, ax = plt.subplots(figsize = (20, 20))
sns.heatmap(house_df.corr(), annot = True)

# Data cleansing
selected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement']

x = house_df[selected_features]
y = house_df['price']

scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# Normalizing output
y = y.values.reshape(-1, 1)
y_scaled = scaler.fit_transform(y)

# Training
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size = 0.25)

# Defining model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100, activation='relu', input_shape=(7, )))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=100, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.summary()

# Compile model
model.compile(optimizer='Adam', loss='mean_squared_error')
epochs_hist = model.fit(x_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.2)

# Evaluating model
epochs_hist.history.keys()

# Graphic
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.title('Progreso del modelo durante Entrenamiento')
plt.xlabel('Epoch')
plt.ylabel('Training and Validation Loss')
plt.legend(['Training Loss', 'Validation Loss'])

# Prediction
## Definir hogar por predecir con sus respectivos inputs / outputs
## 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement'
x_test_1 = np.array([[4, 3, 1960, 5000, 1, 2000, 3000]])

scaler_1 = MinMaxScaler()
x_test_scaled_1 = scaler_1.fit_transform(x_test_1)

# Making predict
y_predict_1 = model.predict(x_test_scaled_1)

# Reinvirtiendo escalado para ver precio normal
y_predict_1 = scaler.inverse_transform(y_predict_1)





