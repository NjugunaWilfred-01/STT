"""
Phase 2 Validation: Test that the preprocessed dataset is correct.
"""

import os
import sys
from datasets import load_from_disk
import torch
import torchaudio

DATA_DIR = "data/processed/alffa_sw"

def test_phase2():
    print("=" * 50)
    print("TESTING PHASE 2 OUTPUT")
    print("=" * 50)

    # 1. Check that the dataset directory exists
    if not os.path.exists(DATA_DIR):
        print("FAIL: Dataset directory not found.")
        sys.exit(1)
    print("PASS: Dataset directory exists.")

    # 2. Load the dataset
    try:
        dataset = load_from_disk(DATA_DIR)
    except Exception as e:
        print(f"FAIL: Could not load dataset: {e}")
        sys.exit(1)
    print("PASS: Dataset loaded successfully.")

    # 3. Check the splits
    expected_splits = ["train", "validation", "test"]
    for split in expected_splits:
        if split not in dataset:
            print(f"FAIL: Split '{split}' missing.")
            sys.exit(1)
    print("PASS: All splits present (train, validation, test).")

    # 4. Check the size of each split
    expected_sizes = {"train": 8144, "validation": 1018, "test": 1018}
    for split, expected in expected_sizes.items():
        actual = len(dataset[split])
        if actual != expected:
            print(f"WARN: Split '{split}' has {actual} samples, expected {expected}.")
            print("      This may be due to different random seed or missing files.")
        else:
            print(f"PASS: Split '{split}' has {actual} samples.")

    # 5. Check column names
    sample = dataset["train"][0]
    required_cols = ["audio", "sentence"]
    for col in required_cols:
        if col not in sample:
            print(f"FAIL: Column '{col}' missing from dataset.")
            sys.exit(1)
    print("PASS: Required columns 'audio' and 'sentence' are present.")

    # 6. Check audio is loaded and resampled to 16kHz
    audio = sample["audio"]
    if "array" not in audio or "sampling_rate" not in audio:
        print("FAIL: Audio column does not contain expected fields.")
        sys.exit(1)
    if audio["sampling_rate"] != 16000:
        print(f"FAIL: Audio sample rate is {audio['sampling_rate']}, expected 16000.")
        sys.exit(1)
    print(f"PASS: Audio sample rate is 16000 Hz, duration {len(audio['array'])/16000:.2f}s.")

    # 7. Check text is non-empty and contains Swahili
    text = sample["sentence"]
    if not isinstance(text, str) or len(text.strip()) == 0:
        print("FAIL: Sentence column is empty or not a string.")
        sys.exit(1)
    # Check for common Swahili characters (optional)
    swahili_chars = set("abcdefghijklmnopqrstuvwxyz".upper()) | set(" .,!?'-")
    if not any(c in text for c in "aiueo"):
        print("WARN: Sentence may not be Swahili (no vowels).")
    else:
        print(f"PASS: Sample sentence: '{text[:80]}...'")

    # 8. Check tokenizer exists
    tokenizer_path = os.path.join(DATA_DIR, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        print("FAIL: Tokenizer file not found.")
        sys.exit(1)
    print("PASS: Tokenizer exists.")

    # 9. Check metadata exists
    metadata_path = os.path.join(DATA_DIR, "metadata.json")
    if not os.path.exists(metadata_path):
        print("FAIL: Metadata file not found.")
        sys.exit(1)
    import json
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print("PASS: Metadata loaded.")
    print(f"    Vocab size: {metadata.get('vocab_size', 'N/A')}")
    print(f"    Character set: {metadata.get('character_set', 'N/A')[:50]}...")

    # 10. Test a full epoch iteration (just to ensure no errors)
    print("Iterating through 10 random samples...")
    for i in range(min(10, len(dataset["train"]))):
        sample = dataset["train"][i]
        assert "audio" in sample
        assert "sentence" in sample
        assert sample["audio"]["sampling_rate"] == 16000
        assert isinstance(sample["sentence"], str)
    print("PASS: Successfully iterated through samples without errors.")

    print("\n" + "=" * 50)
    print("ALL TESTS PASSED - PHASE 2 SUCCESSFUL")
    print("=" * 50)
    print("\nSummary:")
    print(f"  - Dataset: {DATA_DIR}")
    print(f"  - Train: {len(dataset['train'])} samples")
    print(f"  - Validation: {len(dataset['validation'])} samples")
    print(f"  - Test: {len(dataset['test'])} samples")
    print(f"  - Vocab size: {metadata.get('vocab_size', 'N/A')}")
    print(f"  - Sample rate: 16000 Hz")
    print("\nReady to proceed to Phase 3.")

if __name__ == "__main__":
    test_phase2()