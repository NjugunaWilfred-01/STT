"""
Phase 2: Preprocess ALFFA Swahili Dataset
Improved audio file discovery.
"""

import os
import sys
import json
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
from datasets import Dataset, DatasetDict, Audio, Features, Value
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from sklearn.model_selection import train_test_split


# CONFIGURATION

DATA_DIR = "data/raw/alffa_sw"
OUTPUT_DIR = "data/processed/alffa_sw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TOKENIZER_SAVE_PATH = os.path.join(OUTPUT_DIR, "tokenizer.json")
SAMPLE_RATE = 16000
TEST_SIZE = 0.1
VAL_SIZE = 0.1


# STEP 1: LOAD AND PARSE KALDI FILES

print("Loading Kaldi files...")

text_path = os.path.join(DATA_DIR, "text")
wav_scp_path = os.path.join(DATA_DIR, "wav.scp")

if not os.path.exists(text_path) or not os.path.exists(wav_scp_path):
    print("Error: text or wav.scp not found.")
    sys.exit(1)

with open(text_path, 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

with open(wav_scp_path, 'r', encoding='utf-8') as f:
    wav_lines = f.readlines()

text_dict = {}
for line in text_lines:
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        text_dict[parts[0]] = parts[1]

wav_dict = {}
for line in wav_lines:
    parts = line.strip().split(maxsplit=1)
    if len(parts) == 2:
        wav_dict[parts[0]] = parts[1]

common_ids = sorted(set(text_dict.keys()) & set(wav_dict.keys()))
print(f"Found {len(common_ids)} matching utterance IDs.")

if len(common_ids) == 0:
    print("Error: No matching IDs.")
    sys.exit(1)


# STEP 2: BUILD A MAP OF ALL AUDIO FILES IN DATA_DIR

print("Building audio file index...")

audio_extensions = {'.wav', '.flac', '.mp3', '.m4a', '.aiff', '.aif'}
audio_files_by_name = {}  # basename (no ext) -> full path
audio_files_by_fullname = {}  # full filename -> full path

for root, dirs, files in os.walk(DATA_DIR):
    for file in files:
        ext = os.path.splitext(file)[1].lower()
        if ext in audio_extensions:
            full_path = os.path.join(root, file)
            basename = os.path.splitext(file)[0]
            audio_files_by_name[basename] = full_path
            audio_files_by_fullname[file] = full_path

print(f"Found {len(audio_files_by_name)} audio files in {DATA_DIR}")


# STEP 3: RESOLVE AUDIO PATHS USING MULTIPLE STRATEGIES

print("Resolving audio paths...")

audio_paths = []
transcripts = []
missing_count = 0

for utt_id in common_ids:
    raw_path = wav_dict[utt_id]
    found_path = None

    # Strategy 1: Try absolute path as is
    if os.path.exists(raw_path):
        found_path = raw_path
    # Strategy 2: Try relative to DATA_DIR
    elif os.path.exists(os.path.join(DATA_DIR, raw_path)):
        found_path = os.path.join(DATA_DIR, raw_path)
    # Strategy 3: Try basename matching (strip extension if present)
    else:
        # Extract basename without extension from raw_path
        base = os.path.basename(raw_path)
        base_no_ext = os.path.splitext(base)[0]
        
        # Look up in our index
        if base_no_ext in audio_files_by_name:
            found_path = audio_files_by_name[base_no_ext]
        elif base in audio_files_by_fullname:
            found_path = audio_files_by_fullname[base]
        # Strategy 4: Try utterance ID as filename (some datasets use ID)
        elif utt_id in audio_files_by_name:
            found_path = audio_files_by_name[utt_id]
        # Strategy 5: Try common extensions with utt_id
        else:
            for ext in audio_extensions:
                candidate = os.path.join(DATA_DIR, f"{utt_id}{ext}")
                if os.path.exists(candidate):
                    found_path = candidate
                    break
                candidate = os.path.join(DATA_DIR, "wav", f"{utt_id}{ext}")
                if os.path.exists(candidate):
                    found_path = candidate
                    break
                candidate = os.path.join(DATA_DIR, "audio", f"{utt_id}{ext}")
                if os.path.exists(candidate):
                    found_path = candidate
                    break

    if found_path:
        audio_paths.append(found_path)
        transcripts.append(text_dict[utt_id])
    else:
        missing_count += 1

print(f"Successfully resolved {len(audio_paths)} audio files.")
print(f"Missing {missing_count} files (skipped).")

if len(audio_paths) == 0:
    print("\nERROR: No audio files could be found.")
    print("Debug: First 5 lines of wav.scp:")
    for i in range(min(5, len(wav_lines))):
        print(f"  {wav_lines[i].strip()}")
    print("\nDebug: First 5 audio files found in DATA_DIR:")
    for i, (name, path) in enumerate(list(audio_files_by_name.items())[:5]):
        print(f"  {name} -> {path}")
    sys.exit(1)


# STEP 4: SPLIT DATA

print("Splitting data...")

indices = list(range(len(audio_paths)))
train_idx, temp_idx = train_test_split(indices, test_size=(TEST_SIZE + VAL_SIZE), random_state=42)
val_idx, test_idx = train_test_split(temp_idx, test_size=(TEST_SIZE / (TEST_SIZE + VAL_SIZE)), random_state=42)

train_paths = [audio_paths[i] for i in train_idx]
train_texts = [transcripts[i] for i in train_idx]
val_paths = [audio_paths[i] for i in val_idx]
val_texts = [transcripts[i] for i in val_idx]
test_paths = [audio_paths[i] for i in test_idx]
test_texts = [transcripts[i] for i in test_idx]

print(f"Train: {len(train_paths)} samples")
print(f"Validation: {len(val_paths)} samples")
print(f"Test: {len(test_paths)} samples")


# STEP 5: BUILD BPE TOKENIZER

print("Building BPE tokenizer...")

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=5000,
    min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
)

tokenizer.train_from_iterator(train_texts, trainer=trainer)
tokenizer.save(TOKENIZER_SAVE_PATH)
print(f"Tokenizer saved to {TOKENIZER_SAVE_PATH}")
print(f"Vocabulary size: {tokenizer.get_vocab_size()}")


# STEP 6: BUILD DATASET DICT

print("Building Hugging Face DatasetDict...")

def create_dataset(paths, texts):
    data = {
        "audio": paths,
        "sentence": texts
    }
    dataset = Dataset.from_dict(data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    return dataset

train_dataset = create_dataset(train_paths, train_texts)
val_dataset = create_dataset(val_paths, val_texts)
test_dataset = create_dataset(test_paths, test_texts)

dataset_dict = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset,
    "test": test_dataset
})


# STEP 7: SAVE DATASET

print(f"Saving dataset to {OUTPUT_DIR}...")
dataset_dict.save_to_disk(OUTPUT_DIR)
print("Dataset saved successfully.")


# STEP 8: SAVE METADATA

metadata = {
    "dataset": "ALFFA Swahili Broadcast News",
    "total_utterances": len(audio_paths),
    "train_size": len(train_paths),
    "validation_size": len(val_paths),
    "test_size": len(test_paths),
    "sample_rate": SAMPLE_RATE,
    "vocab_size": tokenizer.get_vocab_size(),
    "character_set": sorted(set("".join(transcripts)))
}

with open(os.path.join(OUTPUT_DIR, "metadata.json"), "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 50)
print("PHASE 2 COMPLETE")
print("=" * 50)
print(f"Dataset saved to: {OUTPUT_DIR}")
print(f"Tokenizer saved to: {TOKENIZER_SAVE_PATH}")
print(f"Metadata saved to: {os.path.join(OUTPUT_DIR, 'metadata.json')}")
print("\nNext: Phase 3 - Model Fine-Tuning")