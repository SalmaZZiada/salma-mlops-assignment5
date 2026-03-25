import mlflow
import sys
mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

print("Fetching run from MLflow")
print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")

if accuracy < 0.85:
    print("Model rejected: accuracy below threshold")
    sys.exit(1)
else:
    print("Model accepted: accuracy above threshold")
    sys.exit(0)