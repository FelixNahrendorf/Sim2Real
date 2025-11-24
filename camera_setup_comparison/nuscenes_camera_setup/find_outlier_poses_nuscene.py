import json
from collections import Counter

# Read the JSON file
input_file = "/app/datasets/nuscenes_full/v1.0-trainval/calibrated_sensor.json"
output_file = "nuscenes_15_outlier_poses.txt"

with open(input_file, 'r') as f:
    data = json.load(f)

# Extract all translation[0] values with their tokens
translation_first_values = {}
for entry in data:
    if 'translation' in entry and len(entry['translation']) > 0:
        first_val = entry['translation'][0]
        token = entry['token']
        
        if first_val not in translation_first_values:
            translation_first_values[first_val] = []
        translation_first_values[first_val].append(token)

# Find the value that appears exactly 15 times
target_value = None
target_tokens = []

for value, tokens in translation_first_values.items():
    if len(tokens) == 15:
        target_value = value
        target_tokens = tokens
        print(f"Found value {value} appearing exactly 15 times")
        break

if target_tokens:
    # Write tokens to file
    with open(output_file, 'w') as f:
        for token in target_tokens:
            f.write(f"{token}\n")
    
    print(f"Written {len(target_tokens)} tokens to {output_file}")
    print(f"Translation[0] value: {target_value}")
else:
    print("No translation[0] value appears exactly 15 times")
    
    # Debug: Show the distribution
    print("\nDistribution of translation[0] values:")
    counts = {val: len(tokens) for val, tokens in translation_first_values.items()}
    for val, count in sorted(counts.items(), key=lambda x: x[1]):
        if count < 30:  # Only show values appearing less than 30 times
            print(f"  {val}: {count} times")