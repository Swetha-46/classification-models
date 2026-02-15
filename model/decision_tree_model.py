import joblib
from sklearn.tree import DecisionTreeClassifier


def train(X_train, y_train):
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/decision_tree.pkl")
    return model


def load():
    return joblib.load("model/decision_tree.pkl")