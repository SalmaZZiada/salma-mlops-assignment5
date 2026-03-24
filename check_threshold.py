import mlflow
import sys

mlflow.set_tracking_uri("http://localhost:5000")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Accuracy:", accuracy)

if accuracy < 0.85:
    print("Failed")
    sys.exit(1)
else:
    print("Passed")