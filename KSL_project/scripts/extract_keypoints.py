import os
import subprocess
import pandas as pd
from tqdm import tqdm

DATASET_PATH = "/home/cipher/Downloads/archive"
OUTPUT_PATH = "data/processed"
VIDEO_EXT = ".mov"

WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "extract_worker.py")


def load_splits():
    csv_path = os.path.join(DATASET_PATH, "splits_info.csv")
    
    if not os.path.exists(csv_path):
        print(f"ERROR: splits_info.csv not found at: {csv_path}")
        return None, None, None
    
    df = pd.read_csv(csv_path)
    
    print(f"CSV loaded: {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    train_videos = {}
    val_videos = {}
    test_videos = {}
    
    for idx, row in df.iterrows():
        video_name = row['filename']
        split_name = str(row['split']).strip().lower()
        
        if split_name == 'train':
            train_videos[video_name] = True
        elif split_name == 'val':
            val_videos[video_name] = True
        elif split_name == 'test':
            test_videos[video_name] = True
    
    print(f"Found: {len(train_videos)} train, {len(val_videos)} val, {len(test_videos)} test videos")
    
    return train_videos, val_videos, test_videos


def process_video(video_path, output_path):
    try:
        result = subprocess.run(
            ["python", WORKER_SCRIPT, video_path, output_path],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0 and "FAIL" not in result.stdout:
            return True
        else:
            print(f"  Failed: {os.path.basename(video_path)}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"  Timeout: {os.path.basename(video_path)}")
        return False
    except Exception as e:
        print(f"  Error: {os.path.basename(video_path)} - {e}")
        return False


def process_class_folder(class_name, split_name, video_dict):
    class_input_path = os.path.join(DATASET_PATH, class_name)
    class_output_path = os.path.join(OUTPUT_PATH, split_name, class_name)
    
    if not os.path.exists(class_input_path):
        print(f"  Warning: Class folder not found: {class_name}")
        return
    
    os.makedirs(class_output_path, exist_ok=True)
    
    all_videos = [f for f in os.listdir(class_input_path) 
                  if f.lower().endswith(VIDEO_EXT.lower())]
    
    if len(all_videos) == 0:
        print(f"  Warning: No {VIDEO_EXT} files found in: {class_name}")
        return
    
    split_videos = []
    for vid in all_videos:
        if vid in video_dict:
            split_videos.append(vid)
    
    if len(split_videos) == 0:
        print(f"  Warning: No matching videos in {split_name} for {class_name} (found {len(all_videos)} total)")
        split_videos = all_videos
    
    print(f"  {class_name}: {len(split_videos)} videos for {split_name}")
    
    for video_file in tqdm(split_videos, desc=f"  {class_name} ({split_name})"):
        video_path = os.path.join(class_input_path, video_file)
        output_filename = video_file.replace(VIDEO_EXT, ".npy")
        output_path = os.path.join(class_output_path, output_filename)
        
        process_video(video_path, output_path)


if __name__ == "__main__":
    print("=" * 60)
    print("PHASE 1: KSL Keypoint Extraction (Using splits_info.csv)")
    print("=" * 60)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Video extension: {VIDEO_EXT}")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset path not found: {DATASET_PATH}")
        exit()

    if not os.path.exists(WORKER_SCRIPT):
        print(f"ERROR: Worker script not found: {WORKER_SCRIPT}")
        print("Make sure extract_worker.py is in the same folder.")
        exit()

    train_videos, val_videos, test_videos = load_splits()
    
    if train_videos is None:
        exit()
    
    all_items = os.listdir(DATASET_PATH)
    class_folders = []
    for item in all_items:
        item_path = os.path.join(DATASET_PATH, item)
        if os.path.isdir(item_path):
            has_videos = any(f.lower().endswith(VIDEO_EXT.lower()) 
                           for f in os.listdir(item_path))
            if has_videos:
                class_folders.append(item)
    
    print(f"\nFound {len(class_folders)} class folders with {VIDEO_EXT} files.")
    
    splits = [
        ("train", train_videos),
        ("val", val_videos),
        ("test", test_videos)
    ]
    
    for split_name, video_dict in splits:
        print(f"\n{'='*50}")
        print(f"Processing {split_name.upper()} split ({len(video_dict)} videos total)")
        print(f"{'='*50}")
        
        for class_name in class_folders:
            process_class_folder(class_name, split_name, video_dict)
    
    print("\n" + "=" * 60)
    print("PHASE 1 COMPLETE: Keypoints extracted and saved.")
    print(f"Output saved in: {OUTPUT_PATH}")
    print("=" * 60)
    
    print("\nSummary:")
    for split_name in ["train", "val", "test"]:
        split_path = os.path.join(OUTPUT_PATH, split_name)
        if os.path.exists(split_path):
            total_files = 0
            for root, dirs, files in os.walk(split_path):
                total_files += len([f for f in files if f.endswith('.npy')])
            print(f"  {split_name}: {total_files} .npy files extracted")
        else:
            print(f"  {split_name}: 0 files (folder not created)")