import json
import csv

with open("data/fraud_results_final.json", "r") as f:
    data = json.load(f)

with open("data/fraud_results_final.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=data[0].keys())
    writer.writeheader()
    writer.writerows(data)
