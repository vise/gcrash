# %%ß
# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Create a synthetic dataset for Chirps per Minute vs Temperature in Celsius
np.random.seed(42) # for reproducibility
temp = np.linspace(0, 30, 20) # temperatures from 0 to 30
chirps = 3 * temp + 4 + np.random.normal(0, 3, 20) # chirps = 3*temperature + 4 + noise

# Visualize the data
plt.scatter(temp, chirps)
plt.title('Chirps per Minute vs Temperature')
plt.xlabel('Temperature in Celsius')
plt.ylabel('Chirps per Minute')
plt.show()
# plt.savefig("test.png")
# Reshape the arrays to 2D as required by the algorithm
temp = temp.reshape(-1, 1)
chirps = chirps.reshape(-1, 1)

# Split into training and test set
temp_train, temp_test, chirps_train, chirps_test = train_test_split(temp, chirps, test_size=0.2, random_state=42)

# Create a Linear Regression classifier
model = LinearRegression()

# Fit the model to the data
model.fit(temp_train, chirps_train)

# Predict the labels of the test set
chirps_pred = model.predict(temp_test)

# Plot the regression line
plt.scatter(temp, chirps, color='blue') # plotting the initial data
plt.plot(temp_test, chirps_pred, color='red') # plotting the line
plt.title('Chirps per Minute vs Temperature')
plt.xlabel('Temperature in Celsius')
plt.ylabel('Chirps per Minute')
plt.show()

# %%
