import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="ElectroGuard Research Demo")

st.title("⚡ ElectroGuard – Model Comparison Demo")

st.write("This demo compares Logistic Regression and Random Forest on synthetic electricity data.")

# Generate lightweight synthetic dataset
@st.cache_data
def generate_data():
    np.random.seed(42)
    n_users = 500   # small dataset (IMPORTANT)
    days = 30       # reduced days

    data = np.random.normal(50, 10, (n_users, days))
    theft = np.random.choice([0, 1], n_users, p=[0.85, 0.15])

    for i in range(n_users):
        if theft[i] == 1:
            data[i] *= np.random.uniform(0.3, 0.6)

    df = pd.DataFrame(data)
    df["theft"] = theft
    return df

df = generate_data()

X = df.drop("theft", axis=1)
y = df["theft"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=50)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)
rf_acc = accuracy_score(y_test, rf_pred)

# Display Results
st.subheader("📊 Model Accuracy Comparison")

st.write(f"Logistic Regression Accuracy: **{lr_acc:.2f}**")
st.write(f"Random Forest Accuracy: **{rf_acc:.2f}**")

# Plot
fig, ax = plt.subplots()
models = ["Logistic Regression", "Random Forest"]
accuracies = [lr_acc, rf_acc]
ax.bar(models, accuracies)
ax.set_ylim(0, 1)
ax.set_ylabel("Accuracy")

st.pyplot(fig)

st.success("Demo completed successfully 🚀")
