import mlflow
import os
import sys

with open("accuracy.txt", "r") as f:
    acc = float(f.read())

if acc < 0.85:
    raise Exception(f"Accuracy too low: {acc}")

print(f"Accuracy OK: {acc}")