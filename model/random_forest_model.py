import joblib
from sklearn.ensemble import RandomForestClassifier


def train(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/random_forest.pkl")
    return model


def load():
    return joblib.load("model/random_forest.pkl")
