import ast
import pandas as pd

rows = []
with open("summary_results.txt", "r") as f:
    for line in f:
        try:
            row = ast.literal_eval(line.strip())  # safely parse the list
            rows.append(row)
        except Exception as e:
            print("Skipping line due to error:", e)


df = pd.DataFrame(
    rows,
    columns=[
        "test_loss",
        "test_acc",
        "avg_latency",
        "model_size",
        "train_time",
        "int8_accuracy",
        "params",
    ],
)


print(df.to_string(index=False))

df.to_csv("summary_results.csv", index=False)
df.to_json("summary_results.json", orient="records", indent=2)
