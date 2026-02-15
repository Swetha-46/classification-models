import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

# Import model trainers
from model.logistic_model import train as train_logistic
from model.decision_tree_model import train as train_decision
from model.knn_model import train as train_knn
from model.naive_bayes_model import train as train_nb
from model.random_forest_model import train as train_rf
from model.xgboost_model import train as train_xgb

# Create folders if not exist
os.makedirs("dataset", exist_ok=True)
os.makedirs("model", exist_ok=True)

# Remove old files
files_to_delete = [
    "dataset/test_data.csv",
    "model/scaler.pkl",
    "model/logistic_model.pkl",
    "model/decision_tree.pkl",
    "model/knn_model.pkl",
    "model/naive_bayes.pkl",
    "model/random_forest.pkl",
    "model/xgboost.pkl"
]

for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file)
        print(f"Deleted old file: {file}")

# Load dataset
data = pd.read_csv("dataset/training_dataset.csv")

# Create target column
data["target"] = data["Exam_Score"].apply(lambda x: 1 if x >= 35 else 0)

# Drop original score column
data.drop("Exam_Score", axis=1, inplace=True)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include='object').columns:
    data[col] = le.fit_transform(data[col])

# Split dataset
X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Save fresh test dataset
test_data = X_test.copy()
test_data["target"] = y_test

test_data.to_csv("dataset/test_data.csv", index=False)

print("Test dataset saved successfully")

# Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)

# Save scaler
joblib.dump(scaler, "model/scaler.pkl")

print("Scaler saved successfully")

# Train models
logistic = LogisticRegression()
decision = DecisionTreeClassifier()
knn = KNeighborsClassifier()
nb = GaussianNB()
rf = RandomForestClassifier()
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')


logistic.fit(X_train_scaled, y_train)
decision.fit(X_train_scaled, y_train)
knn.fit(X_train_scaled, y_train)
nb.fit(X_train_scaled, y_train)
rf.fit(X_train_scaled, y_train)
xgb.fit(X_train_scaled, y_train)

# Train all models
train_logistic(X_train, y_train)
train_decision(X_train, y_train)
train_knn(X_train, y_train)
train_nb(X_train, y_train)
train_rf(X_train, y_train)
train_xgb(X_train, y_train)

print("All models saved successfully.")
