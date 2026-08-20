"""
Phase 1: Extract ALFFA Swahili Dataset
Extracts data_broadcastnews_sw.tar.bz2 and organizes the files.
"""

import os
import sys
import tarfile
import shutil

DOWNLOAD_DIR = os.path.expanduser("~/Downloads")
TARGET_DIR = "data/raw/alffa_sw"

archive_path = os.path.join(DOWNLOAD_DIR, "data_broadcastnews_sw.tar.bz2")

if not os.path.exists(archive_path):
    print(f"File not found: {archive_path}")
    print("Please check the download location.")
    sys.exit(1)

print(f"Found archive: {archive_path}")

extract_dir = "data/raw/temp_extract"
os.makedirs(extract_dir, exist_ok=True)

print("Extracting archive (this may take a few minutes)...")

with tarfile.open(archive_path, 'r:bz2') as tar:
    tar.extractall(extract_dir)

print("Extraction complete.")

# Find the actual data folder
for root, dirs, files in os.walk(extract_dir):
    if "wav.scp" in files and "text" in files:
        data_root = root
        break
else:
    print("Could not find wav.scp and text files.")
    print("Listing contents of extraction directory:")
    for root, dirs, files in os.walk(extract_dir):
        print(f"  {root}: {dirs[:5]}, {files[:5]}")
    sys.exit(1)

print(f"Found data at: {data_root}")

if os.path.exists(TARGET_DIR):
    shutil.rmtree(TARGET_DIR)

shutil.move(data_root, TARGET_DIR)
shutil.rmtree(extract_dir)

print(f"Dataset moved to: {TARGET_DIR}")

print("\nExtraction complete.")
print(f"wav.scp: {os.path.join(TARGET_DIR, 'wav.scp')}")
print(f"text: {os.path.join(TARGET_DIR, 'text')}")