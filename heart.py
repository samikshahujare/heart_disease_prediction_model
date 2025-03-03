import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, RocCurveDisplay
from sklearn.preprocessing import StandardScaler

# Streamlit App UI
st.title("Heart Disease Prediction Using Machine Learning")

# File Upload
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Display Dataset Info
    st.write("### Dataset Info")
    st.write(df.describe())

    # Handle Missing Values
    st.write("### Missing Values")
    st.write(df.isna().sum())

    # Data Visualization
    st.write("### Target Column Distribution")
    fig, ax = plt.subplots()
    df["target"].value_counts().plot(kind="bar", color=["salmon", "lightblue"], ax=ax)
    ax.set_title("Heart Disease Frequency")
    ax.set_xlabel("0 = No Disease, 1 = Disease")
    ax.set_ylabel("Count")
    st.pyplot(fig)

    # Data Preprocessing
    X = df.drop("target", axis=1)
    y = df["target"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Selection
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "K-Nearest Neighbors": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    # Train & Evaluate Models
    model_scores = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        model_scores[name] = model.score(X_test_scaled, y_test)

    # Display Model Scores
    st.write("### Model Accuracy Comparison")
    st.write(pd.DataFrame(model_scores, index=["Accuracy"]))

    # Hyperparameter Tuning for Logistic Regression
    if st.button("Perform Hyperparameter Tuning"):
        log_reg_grid = {"C": np.logspace(-4, 4, 20), "solver": ["liblinear"]}
        rs_log_reg = RandomizedSearchCV(LogisticRegression(), param_distributions=log_reg_grid, cv=5, n_iter=20, verbose=0)
        rs_log_reg.fit(X_train_scaled, y_train)
        st.write("### Best Parameters for Logistic Regression")
        st.write(rs_log_reg.best_params_)

    # Predictions & Evaluation
    st.write("### Confusion Matrix")
    best_model = LogisticRegression(C=0.20433597178569418, solver="liblinear")
    best_model.fit(X_train_scaled, y_train)
    y_preds = best_model.predict(X_test_scaled)

    fig, ax = plt.subplots(figsize=(3, 3))
    sns.heatmap(confusion_matrix(y_test, y_preds), annot=True, cmap="Blues", fmt="d", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # Classification Report
    st.write("### Classification Report")
    st.text(classification_report(y_test, y_preds))

    # Additional Metrics
    st.write("### Model Performance Metrics")
    cv_acc = np.mean(cross_val_score(best_model, X, y, cv=5, scoring="accuracy"))
    cv_precision = np.mean(cross_val_score(best_model, X, y, cv=5, scoring="precision"))
    cv_recall = np.mean(cross_val_score(best_model, X, y, cv=5, scoring="recall"))
    cv_f1 = np.mean(cross_val_score(best_model, X, y, cv=5, scoring="f1"))
    
    metrics_df = pd.DataFrame({
        "Accuracy": [cv_acc],
        "Precision": [cv_precision],
        "Recall": [cv_recall],
        "F1 Score": [cv_f1]
    })
    st.write(metrics_df)

    # Feature Importance
    st.write("### Feature Importance")
    feature_importance = pd.DataFrame(best_model.coef_[0], index=X.columns, columns=["Importance"])
    feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
    st.bar_chart(feature_importance)

else:
    st.warning("Please upload a CSV file to continue.")