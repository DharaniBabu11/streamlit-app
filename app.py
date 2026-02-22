import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="ElectroGuard", layout="wide")

st.title("⚡ ElectroGuard")
st.markdown("### AI-Based Electricity Theft Detection System")
st.markdown("Final Year Project – Machine Learning Model Comparison")

st.info("""
👨‍🎓 Student: Your Name  
🎓 Course: B.Tech / B.E  
🏫 College: Your College Name  
📅 Academic Year: 2025–2026  
""")

st.markdown("---")

# -------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------
st.sidebar.header("⚙ Simulation Settings")

n_users = st.sidebar.slider("Number of Users", 200, 1000, 500)
theft_ratio = st.sidebar.slider("Theft Percentage", 0.05, 0.40, 0.15)
random_seed = st.sidebar.number_input("Random Seed", value=42)

run_button = st.sidebar.button("🚀 Run Simulation")

# -------------------------------------------------
# DATA GENERATION FUNCTION
# -------------------------------------------------
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

# -------------------------------------------------
# MAIN EXECUTION
# -------------------------------------------------
if run_button:

    with st.spinner("Training models and evaluating performance..."):

        df = generate_data(n_users, theft_ratio, random_seed)

        X = df.drop("theft", axis=1)
        y = df["theft"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed
        )

        # Models
        lr = LogisticRegression(max_iter=500)
        rf = RandomForestClassifier(n_estimators=100)
        iso = IsolationForest(contamination=theft_ratio, random_state=random_seed)

        lr.fit(X_train, y_train)
        rf.fit(X_train, y_train)
        iso.fit(X_train)

        lr_pred = lr.predict(X_test)
        rf_pred = rf.predict(X_test)

        iso_pred = iso.predict(X_test)
        iso_pred = np.where(iso_pred == -1, 1, 0)

        lr_acc = accuracy_score(y_test, lr_pred)
        rf_acc = accuracy_score(y_test, rf_pred)
        iso_acc = accuracy_score(y_test, iso_pred)

    # -------------------------------------------------
    # TABS
    # -------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "📈 Visual Analysis", "📄 Project Report"])

    # ---------------- Dashboard ----------------
    with tab1:
        st.subheader("Model Performance Overview")

        col1, col2, col3 = st.columns(3)
        col1.metric("Logistic Regression", f"{lr_acc:.2f}")
        col2.metric("Random Forest", f"{rf_acc:.2f}")
        col3.metric("Isolation Forest", f"{iso_acc:.2f}")

        st.markdown("### 📋 Model Comparison Table")

        results = pd.DataFrame({
            "Model": ["Logistic Regression", "Random Forest", "Isolation Forest"],
            "Accuracy": [lr_acc, rf_acc, iso_acc]
        })

        st.dataframe(results, use_container_width=True)

    # ---------------- Visuals ----------------
    with tab2:

        st.subheader("Confusion Matrix – Random Forest")

        cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        st.subheader("ROC Curve – Random Forest")

        rf_probs = rf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, rf_probs)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.plot([0,1], [0,1], linestyle="--")
        ax2.set_xlabel("False Positive Rate")
        ax2.set_ylabel("True Positive Rate")
        ax2.set_title(f"AUC = {roc_auc:.2f}")
        st.pyplot(fig2)

    # ---------------- Report ----------------
    with tab3:

        st.subheader("Classification Report – Random Forest")
        st.text(classification_report(y_test, rf_pred))

        st.markdown("## 🌍 Real-World Impact")

        st.write("""
Electricity theft causes significant revenue losses to power distribution companies.
ElectroGuard leverages machine learning algorithms to detect abnormal consumption 
patterns and assist authorities in identifying potential theft cases.

This system can improve grid reliability, reduce financial losses, 
and enhance monitoring efficiency.
""")

else:
    st.warning("Please click 'Run Simulation' from the sidebar to start the analysis.")
