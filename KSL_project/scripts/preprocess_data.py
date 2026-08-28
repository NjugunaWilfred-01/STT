import os
import numpy as np
import json
from tqdm import tqdm
import sys

# Configuration
PROCESSED_PATH = "data/processed"
OUTPUT_PATH = "data"
SEQUENCE_LENGTH = 30


def load_label_mapping():
    """Load label to integer mapping from JSON file."""
    mapping_path = "data/class_mapping.json"
    
    if not os.path.exists(mapping_path):
        print(f"ERROR: {mapping_path} not found.")
        print("Run create_label_mapping.py first.")
        sys.exit(1)
    
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    
    label_to_int = mapping["label_to_int"]
    int_to_label = mapping["int_to_label"]
    
    print(f"Loaded {len(label_to_int)} classes")
    return label_to_int, int_to_label


def normalize_keypoints(keypoints):
    """
    Normalize keypoints by subtracting shoulder center coordinates.
    This makes the model position-invariant.
    """
    # Pose landmarks are the first 99 values (33 landmarks * 3)
    # Left shoulder is at index 11, right shoulder at index 12
    left_shoulder = keypoints[:, 11*3:11*3+3]
    right_shoulder = keypoints[:, 12*3:12*3+3]
    
    # Center of shoulders (shape: frames, 3)
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    
    # Create a copy and subtract shoulder_center from each triplet
    normalized = keypoints.copy()
    
    # Subtract shoulder center from each (x,y,z) triplet
    for i in range(0, normalized.shape[1], 3):
        normalized[:, i:i+3] = normalized[:, i:i+3] - shoulder_center
    
    return normalized


def pad_or_truncate(sequence, target_length):
    """
    Pad or truncate a sequence to target_length.
    """
    current_length = sequence.shape[0]
    
    if current_length >= target_length:
        # Truncate: take the first target_length frames
        return sequence[:target_length]
    else:
        # Pad: repeat the last frame until target_length
        padding = np.tile(sequence[-1:], (target_length - current_length, 1))
        return np.vstack([sequence, padding])


def load_class_data(split_name, class_name, label_to_int):
    """
    Load all .npy files for a specific class and split.
    """
    class_path = os.path.join(PROCESSED_PATH, split_name, class_name)
    
    if not os.path.exists(class_path):
        return [], []
    
    npy_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
    
    if len(npy_files) == 0:
        return [], []
    
    sequences = []
    labels = []
    
    for npy_file in npy_files:
        file_path = os.path.join(class_path, npy_file)
        
        try:
            data = np.load(file_path)
        except Exception as e:
            print(f"  Warning: Could not load {file_path}: {e}")
            continue
        
        # Skip empty or invalid data
        if data.shape[0] == 0:
            continue
        
        # Normalize keypoints (subtract shoulder center)
        normalized_data = normalize_keypoints(data)
        
        # Pad or truncate to fixed length
        fixed_sequence = pad_or_truncate(normalized_data, SEQUENCE_LENGTH)
        
        sequences.append(fixed_sequence)
        labels.append(label_to_int[class_name])
    
    return sequences, labels


def load_split(split_name, label_to_int, class_folders):
    """
    Load all data for a specific split.
    """
    split_path = os.path.join(PROCESSED_PATH, split_name)
    
    if not os.path.exists(split_path):
        print(f"  Warning: {split_name} folder not found")
        return [], []
    
    all_sequences = []
    all_labels = []
    
    for class_name in tqdm(class_folders, desc=f"  Loading {split_name}"):
        if class_name not in label_to_int:
            print(f"  Warning: {class_name} not in label mapping")
            continue
        
        sequences, labels = load_class_data(split_name, class_name, label_to_int)
        all_sequences.extend(sequences)
        all_labels.extend(labels)
    
    return all_sequences, all_labels


def apply_augmentation(sequences, labels):
    """
    Apply data augmentation to the training data.
    Returns augmented sequences and labels (doubled).
    """
    augmented_sequences = []
    augmented_labels = []
    
    for seq, label in zip(sequences, labels):
        # Original sequence
        augmented_sequences.append(seq)
        augmented_labels.append(label)
        
        # Horizontal flip (mirror x-coordinates)
        # x coordinates are at indices 0, 3, 6, ... (every 3rd value)
        seq_flipped = seq.copy()
        for i in range(0, seq_flipped.shape[1], 3):
            seq_flipped[:, i] = 1.0 - seq_flipped[:, i]
        
        augmented_sequences.append(seq_flipped)
        augmented_labels.append(label)
    
    return augmented_sequences, augmented_labels


def main():
    print("=" * 60)
    print("PHASE 2: Data Preprocessing & Augmentation")
    print("=" * 60)
    
    # Load label mapping
    label_to_int, int_to_label = load_label_mapping()
    num_classes = len(label_to_int)
    print(f"Number of classes: {num_classes}")
    
    # Get all class folders from train split
    train_path = os.path.join(PROCESSED_PATH, "train")
    
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found.")
        print("Run Phase 1 (extract_keypoints.py) first.")
        sys.exit(1)
    
    class_folders = [f for f in os.listdir(train_path) 
                     if os.path.isdir(os.path.join(train_path, f))]
    
    print(f"Found {len(class_folders)} class folders")
    
    # Process each split
    splits = ["train", "val", "test"]
    all_data = {}
    
    for split_name in splits:
        print(f"\nProcessing {split_name} split...")
        sequences, labels = load_split(split_name, label_to_int, class_folders)
        
        if len(sequences) == 0:
            print(f"  Warning: No data loaded for {split_name}")
            all_data[split_name] = ([], [])
            continue
        
        # Apply augmentation only to training data
        if split_name == "train":
            print(f"  Original training samples: {len(sequences)}")
            sequences, labels = apply_augmentation(sequences, labels)
            print(f"  Augmented training samples: {len(sequences)}")
        
        all_data[split_name] = (sequences, labels)
    
    # Convert to numpy arrays and save
    print("\n" + "=" * 60)
    print("Saving processed data...")
    print("=" * 60)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    for split_name, (sequences, labels) in all_data.items():
        if len(sequences) == 0:
            print(f"  {split_name}: No data to save")
            continue
        
        X = np.array(sequences, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        X_path = os.path.join(OUTPUT_PATH, f"X_{split_name}.npy")
        y_path = os.path.join(OUTPUT_PATH, f"y_{split_name}.npy")
        
        np.save(X_path, X)
        np.save(y_path, y)
        
        print(f"  {split_name}: {X.shape} -> {X_path}")
        print(f"  {split_name}: {y.shape} -> {y_path}")
    
    # Save class names
    class_names_path = os.path.join(OUTPUT_PATH, "class_names.txt")
    with open(class_names_path, "w") as f:
        for idx in range(num_classes):
            f.write(f"{idx}: {int_to_label[str(idx)]}\n")
    print(f"\nClass names saved to: {class_names_path}")
    
    # Save shape info
    shape_info = {
        "sequence_length": SEQUENCE_LENGTH,
        "num_features": 1629,
        "num_classes": num_classes,
        "class_names": [int_to_label[str(i)] for i in range(num_classes)]
    }
    
    with open(os.path.join(OUTPUT_PATH, "shape_info.json"), "w") as f:
        json.dump(shape_info, f, indent=2)
    print(f"Shape info saved to: {OUTPUT_PATH}/shape_info.json")
    
    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Output saved in: {OUTPUT_PATH}")
    print(f"Files created:")
    print(f"  - X_train.npy, y_train.npy")
    print(f"  - X_val.npy, y_val.npy")
    print(f"  - X_test.npy, y_test.npy")
    print(f"  - class_names.txt")
    print(f"  - shape_info.json")


if __name__ == "__main__":
    main()