import os
import numpy as np

BASE_PATH = "data/processed"

def verify_split(split_name):
    split_path = os.path.join(BASE_PATH, split_name)
    
    if not os.path.exists(split_path):
        print(f"{split_name}: Folder not found")
        return 0, 0, 0
    
    total_files = 0
    empty_files = 0
    invalid_files = 0
    class_counts = {}
    sample_info = None
    
    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        class_counts[class_name] = len(npy_files)
        total_files += len(npy_files)
        
        # Check a sample file
        for npy_file in npy_files[:3]:
            file_path = os.path.join(class_path, npy_file)
            try:
                data = np.load(file_path)
                if data.shape[0] == 0:
                    empty_files += 1
                if sample_info is None and data.shape[0] > 0:
                    sample_info = (class_name, npy_file, data.shape)
            except Exception:
                invalid_files += 1
    
    print(f"\n{split_name.upper()} Split:")
    print(f"  Total files: {total_files}")
    print(f"  Classes: {len(class_counts)}")
    print(f"  Empty files: {empty_files}")
    print(f"  Invalid files: {invalid_files}")
    if sample_info:
        print(f"  Sample: {sample_info[0]}/{sample_info[1]} -> Shape: {sample_info[2]}")
    
    return total_files, len(class_counts), invalid_files

print("=" * 50)
print("VERIFYING EXTRACTED DATA")
print("=" * 50)

train_total, train_classes, train_invalid = verify_split("train")
val_total, val_classes, val_invalid = verify_split("val")
test_total, test_classes, test_invalid = verify_split("test")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Train: {train_total} files, {train_classes} classes, {train_invalid} invalid")
print(f"Val:   {val_total} files, {val_classes} classes, {val_invalid} invalid")
print(f"Test:  {test_total} files, {test_classes} classes, {test_invalid} invalid")
print("=" * 50)