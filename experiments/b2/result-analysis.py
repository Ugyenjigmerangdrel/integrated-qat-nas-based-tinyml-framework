import json
import pandas as pd

# Load the JSON file
with open("random_nas_summary_results.json", "r") as f:
    data = json.load(f)

# Flatten into a list of rows
rows = []
for model_id, values in data.items():
    test_loss, test_acc, latency, size_kb, train_time, cfg = values
    row = {
        "Model_ID": int(model_id),
        "Test_Loss": test_loss,
        "Test_Accuracy": test_acc,
        "Avg_Latency (s)": latency,
        "Model_Size (KB)": size_kb,
        "Train_Time (s)": train_time,
        **cfg  # unpack architecture parameters
    }
    rows.append(row)

# Create DataFrame
df = pd.DataFrame(rows)

# Sort by accuracy descending
df = df.sort_values(by="Test_Accuracy", ascending=False).reset_index(drop=True)

# Round numeric columns for readability
numeric_cols = ["Test_Loss", "Test_Accuracy", "Avg_Latency (s)", "Model_Size (KB)", "Train_Time (s)"]
df[numeric_cols] = df[numeric_cols].round(4)

# Display the table
print(df.to_string(index=False))
