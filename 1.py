import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0,8,5,2,1])
y_train = np.array([300.0, 500,100,400,500,800])
print(f"x_train = {x_train}")
print(f"y_train = {y_train}")
# Plot the data points
plt.scatter(x_train, y_train,)
# Set the title
plt.title("Housing Prices")
# Set the y-axis label
plt.ylabel('Price (in 1000s of dollars)')
# Set the x-axis label
plt.xlabel('Size (1000 sqft)')
plt.show()
