import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

st.set_page_config(page_title="ElectroGuard - FYP", layout="wide")

st.title("⚡ ElectroGuard: Electricity Theft Detection")
st.markdown("### Final Year Project - Machine Learning Model Comparison")

# ---------------- Sidebar Controls ----------------
st.sidebar.header("Simulation Settings")

n_users = st.sidebar.slider("Number of Users", 200, 1000, 500)
theft_ratio = st.sidebar.slider("Theft Percentage", 0.05, 0.40, 0.15)
random_seed = st.sidebar.number_input("Random Seed", value=42)

# ---------------- Data Generation ----------------
@st.cache_data
def generate_data(n_users, theft_ratio, seed):
    np.random.seed(seed)
    days = 30

    data = np.random.normal(50, 10, (n_users, days))
    theft = np.random.choice([0, 1], n_users, p=[1-theft_ratio, theft_ratio])

    for i in range(n_users):
        if theft[i] == 1:
            data[i] *= np.random.uniform(0.3, 0.6)

    df = pd.DataFrame(data)
    df["theft"] = theft
    return df

df = generate_data(n_users, theft_ratio, random_seed)

X = df.drop("theft", axis=1)
y = df["theft"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed)

# ---------------- Models ----------------
lr = LogisticRegression(max_iter=500)
rf = RandomForestClassifier(n_estimators=100)
iso = IsolationForest(contamination=theft_ratio, random_state=random_seed)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)
iso.fit(X_train)

# Predictions
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

# Convert Isolation Forest output (-1,1) to (1,0)
iso_pred = iso.predict(X_test)
iso_pred = np.where(iso_pred == -1, 1, 0)

# ---------------- Metrics ----------------
st.subheader("📊 Model Performance Comparison")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Logistic Regression Accuracy", f"{accuracy_score(y_test, lr_pred):.2f}")

with col2:
    st.metric("Random Forest Accuracy", f"{accuracy_score(y_test, rf_pred):.2f}")

with col3:
    st.metric("Isolation Forest Accuracy", f"{accuracy_score(y_test, iso_pred):.2f}")

# ---------------- Confusion Matrix ----------------
st.subheader("🔎 Confusion Matrix - Random Forest")

cm = confusion_matrix(y_test, rf_pred)
fig, ax = plt.subplots()
ax.imshow(cm)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

# ---------------- ROC Curve ----------------
st.subheader("📈 ROC Curve - Random Forest")

rf_probs = rf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, rf_probs)
roc_auc = auc(fpr, tpr)

fig2, ax2 = plt.subplots()
ax2.plot(fpr, tpr)
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title(f"ROC Curve (AUC = {roc_auc:.2f})")
st.pyplot(fig2)

# ---------------- Report ----------------
st.subheader("📄 Classification Report - Random Forest")
st.text(classification_report(y_test, rf_pred))

# ---------------- Project Summary ----------------
st.markdown("---")
st.markdown("### 📌 Project Summary")
st.write("""
ElectroGuard uses machine learning techniques to detect abnormal electricity consumption patterns 
that may indicate theft. This project compares supervised learning models and anomaly detection 
methods to evaluate detection performance.
""")
