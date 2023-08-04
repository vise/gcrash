# Importing necessary libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic dataset
X = np.linspace(-10, 10, 1000)
y = X**2 + np.random.normal(0, 5, 1000) # Quadratic function with noise

# Splitting into training and testing sets
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Reshaping the arrays to 2D as required by the algorithm
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Building a neural network model with ReLU activation
model = Sequential([
    Dense(32, input_shape=(1,), activation='relu'),
    Dense(16, activation='relu'),
    Dense(1) # No activation function in the output layer for regression problem
])

# Compile the model with Mean Squared Error loss and SGD optimizer
model.compile(optimizer=SGD(learning_rate=0.01), loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

# Plot the training loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
