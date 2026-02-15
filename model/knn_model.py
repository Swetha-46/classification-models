import joblib
from sklearn.neighbors import KNeighborsClassifier


def train(X_train, y_train):
    model = KNeighborsClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, "model/knn_model.pkl")
    return model


def load():
    return joblib.load("model/knn_model.pkl")
