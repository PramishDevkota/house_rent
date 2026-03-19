import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="House Rent Predictor", layout="centered")

st.title("🏠 House Rent Prediction")
st.write("Enter land size to estimate expected rent.")

# Dataset (Hidden from user)
x = np.array([1.5, 2.0, 2.5, 3.0, 4.0])
y = np.array([150, 200, 250, 300, 400])

# Train model internally

w = 0
b = 0
lr = 0.001
epochs = 4000
n = len(x)

for i in range(epochs):
    y_hat = w * x + b
    dw = (-2/n) * np.sum(x * (y - y_hat))
    db = (-2/n) * np.sum((y - y_hat))
    w -= lr * dw
    b -= lr * db


# User Input

land_size = st.number_input(
    "Enter Land Size (in thousand sq ft)",
    min_value=1.0,
    max_value=5.0,
    value=2.5
)

# Prediction

if st.button("Predict Rent"):

    predicted_rent = w * land_size + b

    st.success(f"Estimated Rent: {round(predicted_rent,2)} Lakhs")

    # Plot
    y_pred = w * x + b

    fig, ax = plt.subplots()
    ax.scatter(x, y, label="Actual Rent")
    ax.plot(x, y_pred, label="Trend Line")
    ax.scatter(land_size, predicted_rent)
    ax.set_xlabel("Land Size")
    ax.set_ylabel("Rent (Lakhs)")
    ax.legend()

    st.pyplot(fig)