import joblib
from xgboost import XGBClassifier


def train(X_train, y_train):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, "model/xgboost.pkl")
    return model


def load():
    return joblib.load("model/xgboost.pkl")
