import matplotlib.pyplot as plt
import numpy as np

x = np.array ([1.5, 2.0, 2.5, 3.0, 4.0, np.inf])  # Land sizes in sq ft
y = np.array([150, 200, 250, 300, 400, ])  # rents in lakhs

w = 0
b = 0
lr = 0.001
n = len(x)
epochs = 4000

for i in range(epochs):
    y_hat = w * x + b


    #gradients
    dw = (-2/n) * np.sum(x * (y - y_hat))
    db = (-2/n) * np.sum((y - y_hat))

    #update parameters 
    w = w - lr * dw
    b = b - lr * db

print(f"Learned slope(w): {w: .2f}, intercept (b): {b: .2f}")

# predictions

y_pred = w * x + b
print("Prediction", y_pred)

#Mean squared Error
mse = np.mean((y - y_pred)**2)
print(f"Mean Squared Error: {mse: .2f}")

#plotting
plt.scatter(x , y, color = 'red', label = 'Actual rent')
plt.plot(x, y_pred)
plt.xlabel('Land Size (sq ft)')
plt.ylabel('Price (Lakhs)')
plt.title("House Rent Prediction using Linear Regression")
plt.legend()
plt.show()
