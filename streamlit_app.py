import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# Page Config
st.set_page_config(
    page_title="ML Classification Dashboard",
    layout="wide"
)

#Custom CSS
st.markdown("""
<style>

/* Main container width */
.block-container {
    max-width: 1000px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Header */
.header {
    background-color: #1f77b4;
    padding: 18px;
    border-radius: 10px;
    color: white;
    text-align: center;
    font-size: 26px;
    font-weight: bold;
    max-width: 800px;
    margin: auto;
}

/* Card */
.card {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

/* Metrics table narrower */
.metrics-table {
    width: 70%;
    margin: auto;
}

/* Confusion matrix center */
.confusion-container {
    width: 60%;
    margin: auto;
}

</style>
""", unsafe_allow_html=True)

#Header
st.markdown('<div class="header">Machine Learning Classification Models Dashboard</div>', unsafe_allow_html=True)
st.write("")

st.subheader("Upload Test Dataset")
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Model Selection")
model_list = [
    "Select a model",
    "Logistic Regression",
    "Decision Tree Classifier",
    "K-Nearest Neighbor Classifier",
    "Naive Bayes Classifier - Gaussian",
    "Ensemble Model - Random Forest",
    "Ensemble Model - XGBoost"
    ]

model_name = st.selectbox(
    "Choose Model",
    model_list,
    index=0
)
st.markdown('</div>', unsafe_allow_html=True)

st.subheader("Model Results")

if model_name == "Select a model":
    st.info("Please select a model.")

elif uploaded_file is None:
    st.warning("Please upload test dataset.")

else:
    data = pd.read_csv(uploaded_file)
    if "target" not in data.columns:
        st.error("Dataset must contain a column named 'target'")
    else:
        X = data.drop("target", axis=1)
        y = data["target"]

        model_paths = {
            "Logistic Regression": "model/logistic_model.pkl",
            "Decision Tree Classifier": "model/decision_tree.pkl",
            "K-Nearest Neighbor Classifier": "model/knn_model.pkl",
            "Naive Bayes Classifier - Gaussian": "model/naive_bayes.pkl",
            "Ensemble Model - Random Forest": "model/random_forest.pkl",
            "Ensemble Model - XGBoost": "model/xgboost.pkl"
        }

        model = joblib.load(model_paths[model_name])

        y_pred = model.predict(X)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)[:, 1]
            auc_score = roc_auc_score(y, y_prob)
        else:
            auc_score = roc_auc_score(y, y_pred)

        #Metrics Table
        metrics_df = pd.DataFrame({
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "MCC",
                "AUC"
            ],
            "Value": [
                round(accuracy_score(y, y_pred), 4),
                round(precision_score(y, y_pred), 4),
                round(recall_score(y, y_pred), 4),
                round(f1_score(y, y_pred), 4),
                round(matthews_corrcoef(y, y_pred), 4),
                round(auc_score, 4)
            ]
        })

        st.markdown("### Evaluation Metrics")
        st.markdown('<div class="metrics-table">', unsafe_allow_html=True)
        st.table(metrics_df.set_index("Metric"))
        st.markdown('</div>', unsafe_allow_html=True)

        #Confusion Matrix
        st.markdown('<div class="confusion-container">', unsafe_allow_html=True)
        with st.spinner("Generating Confusion Matrix... Please wait ⏳"):
            time.sleep(1)
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap="Blues",
                ax=ax,
                annot_kws={"size": 12}
            )
            ax.set_xlabel("Predicted", fontsize=11)
            ax.set_ylabel("Actual", fontsize=11)
            ax.set_title("Confusion Matrix", fontsize=13)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

#Footer
st.markdown("""
<hr>
<center style="color: gray;">
Machine Learning Classification Models Dashboard
</center>
""", unsafe_allow_html=True)
