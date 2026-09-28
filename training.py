import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Conv1D, BatchNormalization, Dense, Dropout
import numpy as np
import matplotlib.pyplot as plt

x_train = np.load('x_train.npy')
y_train = np.load('y_train.npy')
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

print("x_train shape:", x_train.shape)

input_shape=x_train.shape[1:]

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(LSTM(units=64, return_sequences=False))
model.add(Dense(units=1, activation='relu'))

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

history = model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))   

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


plt.figure()
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.grid(True)
plt.show()

model.save('weather_model.h5')