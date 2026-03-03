# Copyright (C) 2025 co-pace GmbH (subsidiary of Continental AG).
# Licensed under the BSD-3-Clause License.
# @author: Marius Kästingschäfer and Théo Gieruc
# ==============================================================================
import sys, os
import numpy as np
import argparse
import json
from tqdm import tqdm

def get_transform_files(data_dir):
    transform_files = []
    for root, dirs, files in os.walk(data_dir):
        transform_files += [
            os.path.join(root, file)
            for file in files
            if file == "transforms_ego.json"
        ]
    print(f"[DEBUG] Found {len(transform_files)} transform files in '{data_dir}':")
    for f in transform_files:
        print(f"  {f}")
    return transform_files

def split_dataset(data_dir, split_ratio):
    transform_files = get_transform_files(data_dir)
    num_files = len(transform_files)
    pbar = tqdm(transform_files, total=num_files)
    
    for file in pbar:
        
        if 'lidar' in file or 'nuscenes' in file:
            #print(f"[DEBUG] Skipping (lidar/invisible): {file}")
            continue
        #if 'nuscenes' in file:
        #    num_frames = 7
        #    indices = np.arange(num_frames)
        #    np.random.shuffle(indices)
        #    num_train_frames = int(num_frames * split_ratio)
        #    train_indices = indices[:num_train_frames]
        #    test_indices = indices[num_train_frames:]
        elif 'sphere' in file:
            indices = np.arange(70, 100)
            np.random.shuffle(indices)
            train_indices = indices[:24]
            test_indices = indices[24:]
        else:
            print(f"[DEBUG] Skipping (no matching condition - not nuscenes/sphere): {file}")
            continue

        print(f"[DEBUG] Processing: {file}")
        print(f"[DEBUG]   train_indices: {sorted(train_indices)}")
        print(f"[DEBUG]   test_indices:  {sorted(test_indices)}")

        with open(file, "r") as f:
            transforms = json.load(f)
        
        frames = transforms["frames"]
        print(f"[DEBUG]   total frames in file: {len(frames)}")

        train = transforms.copy()
        test = transforms.copy()
        train["frames"] = [frames[i] for i in train_indices]
        test["frames"] = [frames[i] for i in test_indices]

        train_file = file.replace(".json", "_BEV70-99_train.json")
        test_file = file.replace(".json", "_BEV70-99_test.json")

        print(f"[DEBUG]   Writing train -> {train_file}")
        print(f"[DEBUG]   Writing test  -> {test_file}")
    
        with open(train_file, "w") as f:
            json.dump(train, f, indent=4)
        with open(test_file, "w") as f:
            json.dump(test, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    np.random.seed(args.seed)
    split_dataset(args.data_dir, args.split_ratio)

# Example: python3.8 split_dataset_BEV.py --data_dir /seed4d/data/static
# python split_dataset_BEV.py --data_dir /app/felix/data/seed4d/data/data_diverse_1600x900_2poses_secogan2/static