import joblib
from sklearn.linear_model import LogisticRegression


def train(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/logistic_model.pkl")
    return model


def load():
    return joblib.load("model/logistic_model.pkl")
