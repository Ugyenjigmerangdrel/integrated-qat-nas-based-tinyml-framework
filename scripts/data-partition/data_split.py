import os 
import sys


data_path = ""

if len(sys.argv) > 1:
    data_path = sys.argv[1]
    print(data_path)
else:
    print("No arguments provided.")
    exit()

with open(os.path.join(data_path, "validation_list.txt")) as f:
    validations_files = set(line.strip() for line in f)

print("Length of Validation Files:", len(validations_files))

with open(os.path.join(data_path, "testing_list.txt")) as f:
    testing_files = set(line.strip() for line in f)
    
print("Length of Testing Files:", len(testing_files))


train_files, validation_files, test_files = [], [], []

for root, _, files in os.walk(data_path):
    for file in files:
        if not file.endswith(".wav"):
            continue

        rel_path = os.path.relpath(os.path.join(root, file), data_path).replace("\\", "/")
        
        if rel_path in validations_files:
            validation_files.append(rel_path)
        elif rel_path in testing_files:
            test_files.append(rel_path)
        else:
            train_files.append(rel_path)


# Write lists for convenience
with open("train_files.txt", "w") as f:
    f.write("\n".join(train_files))
with open("val_files.txt", "w") as f:
    f.write("\n".join(validation_files))
with open("test_files.txt", "w") as f:
    f.write("\n".join(test_files))

print(f"Train: {len(train_files)} files")
print(f"Val:   {len(validation_files)} files")
print(f"Test:  {len(test_files)} files")