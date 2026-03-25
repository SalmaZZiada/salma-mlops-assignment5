import mlflow

mlflow.set_tracking_uri("file:./mlruns")

with open("model_info.txt", "r") as f:
    run_id = f.read().strip()

client = mlflow.tracking.MlflowClient()
run = client.get_run(run_id)

accuracy = run.data.metrics["accuracy"]

if accuracy < 0.85:
    raise Exception(f"Accuracy too low: {accuracy}")

print(f"Accuracy OK: {accuracy}")