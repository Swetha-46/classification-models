import joblib
from sklearn.naive_bayes import GaussianNB


def train(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/naive_bayes.pkl")
    return model


def load():
    return joblib.load("model/naive_bayes.pkl")
