import sys

# نقرأ القيم
with open("accuracy.txt", "r") as f:
    accuracy = float(f.read())

with open("model_info.txt", "r") as f:
    run_id = f.read()

print("Fetching run from MLflow")
print(f"Run ID: {run_id}")
print(f"Accuracy: {accuracy}")


if accuracy < 0.85:
    print("Model rejected: accuracy below threshold")
    sys.exit(1)
else:
    print("Model accepted: accuracy above threshold")
    sys.exit(0)