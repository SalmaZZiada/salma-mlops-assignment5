import mlflow
import os
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Use secret
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

X, y = make_classification(n_samples=500, n_features=10)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

with mlflow.start_run() as run:
    mlflow.log_metric("accuracy", accuracy)

    print("Accuracy:", accuracy)

    with open("model_info.txt", "w") as f:
        f.write(run.info.run_id)