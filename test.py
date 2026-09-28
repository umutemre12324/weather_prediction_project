import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf
from sklearn.metrics import mean_squared_error, root_mean_squared_error

x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')

model=tf.keras.models.load_model('weather_model.h5')

y_pred = model.predict(x_test)
y_pred = y_pred.flatten()

mea=mean_squared_error(y_test, y_pred)
rmse=root_mean_squared_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mea}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


plt.figure()

plt.plot(y_test[:200], label='Actual', alpha=0.7)
plt.plot(y_pred[:200], label='Predicted', alpha=0.7)

plt.title('Actual vs Predicted Temperature - First 200 Hours')
plt.xlabel('Hour')
plt.ylabel('Temperature (C)')

plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(y_test, label='Actual', alpha=0.5)
plt.plot(y_pred, label='Predicted', alpha=0.5)
plt.title('Actual vs Predicted Temperature (C) Over Time')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (C)')
plt.legend()
plt.grid(True)
plt.show()  
