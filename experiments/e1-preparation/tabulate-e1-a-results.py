import ast
import pandas as pd

rows = []
with open("summary_results.txt", "r") as f:
    for line in f:
        try:
            # Parse each line (stringified list)
            row = ast.literal_eval(line.strip())

            # Split metrics and params
            test_loss, test_acc, avg_latency, model_size, train_time, int8_accuracy, params = row

            # Flatten: merge top-level metrics and dict keys
            flat = {
                "test_loss": test_loss,
                "test_acc": test_acc,
                "avg_latency": avg_latency,
                "model_size": model_size,
                "train_time": train_time,
                "int8_accuracy": int8_accuracy,
            }
            flat.update(params)  # expand params dict into columns
            rows.append(flat)
        except Exception as e:
            print("Skipping line due to error:", e)

# Create DataFrame
df = pd.DataFrame(rows)

# Sort by int8 accuracy descending (optional)
df = df.sort_values("int8_accuracy", ascending=False).reset_index(drop=True)

# Display nicely
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 160)
print(df.to_string(index=False))

# Save outputs
df.to_csv("summary_results_parsed.csv", index=False)
df.to_json("summary_results_parsed.json", orient="records", indent=2)
