"""
Phase 3: Fine-Tune wav2vec 2.0 on ALFFA Swahili Data
Loads the preprocessed dataset and fine-tunes a pre-trained model.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Any

import torch
from datasets import load_from_disk
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
import jiwer


# CONFIGURATION

DATA_DIR = "data/processed/alffa_sw"
OUTPUT_DIR = "models/checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
BATCH_SIZE = 8
GRADIENT_ACCUMULATION = 2
EPOCHS = 20
LEARNING_RATE = 3e-4
NUM_WARMUP_STEPS = 100


# STEP 1: LOAD DATASET

print("Loading preprocessed dataset...")
dataset = load_from_disk(DATA_DIR)
print(f"Train: {len(dataset['train'])} samples")
print(f"Validation: {len(dataset['validation'])} samples")
print(f"Test: {len(dataset['test'])} samples")


# STEP 2: LOAD TOKENIZER AND PROCESSOR

print("Loading tokenizer and processor...")

tokenizer_path = os.path.join(DATA_DIR, "tokenizer.json")
tokenizer = Wav2Vec2CTCTokenizer(
    tokenizer_file=tokenizer_path,
    unk_token="[UNK]",
    pad_token="[PAD]",
    bos_token="[CLS]",
    eos_token="[SEP]",
    mask_token="[MASK]",
)

feature_extractor = Wav2Vec2FeatureExtractor(
    feature_size=1,
    sampling_rate=16000,
    padding_value=0.0,
    do_normalize=True,
    return_attention_mask=True,
)

processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor,
    tokenizer=tokenizer,
)

vocab_size = len(processor.tokenizer)
print(f"Vocabulary size: {vocab_size}")


# STEP 3: LOAD MODEL

print("Loading model...")
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_NAME,
    vocab_size=vocab_size,
    mask_time_prob=0.05,
    mask_time_length=10,
    gradient_checkpointing=True,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
)

print(f"Model parameters: {model.num_parameters():,}")


# STEP 4: PREPARE DATA COLLATOR

@dataclass
class DataCollatorWithPadding:
    processor: Any
    padding: bool = True

    def __call__(self, features):
        input_features = []
        labels = []

        for feature in features:
            input_features.append({"input_values": feature["input_values"]})
            labels.append(feature["labels"])

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.tokenizer.pad(
                {"input_ids": labels},
                padding=self.padding,
                return_tensors="pt",
            )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["input_ids"] == self.processor.tokenizer.pad_token_id, -100
        )

        return batch

data_collator = DataCollatorWithPadding(processor=processor)


# STEP 5: PREPARE DATASET
 
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

    return batch

print("Preparing datasets...")
train_dataset = dataset["train"].map(prepare_dataset, remove_columns=dataset["train"].column_names)
val_dataset = dataset["validation"].map(prepare_dataset, remove_columns=dataset["validation"].column_names)
test_dataset = dataset["test"].map(prepare_dataset, remove_columns=dataset["test"].column_names)

 
# STEP 6: EVALUATION METRICS
 
wer_metric = jiwer.wer

def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = torch.argmax(torch.tensor(pred_logits), dim=-1)
    
    pred_str = processor.batch_decode(pred_ids)
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(label_ids)

    wer = wer_metric(label_str, pred_str)
    return {"wer": wer}


# STEP 7: TRAINING ARGUMENTS
 
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    warmup_steps=NUM_WARMUP_STEPS,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    report_to="wandb",
    run_name="alffa_swahili_stt",
    fp16=True,
    dataloader_num_workers=4,
    save_total_limit=3,
)

 
# STEP 8: TRAINER
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

 
# STEP 9: START TRAINING
 
print("Starting training...")
trainer.train()


# STEP 10: EVALUATE ON TEST SET

print("Evaluating on test set...")
test_results = trainer.evaluate(test_dataset)
print(f"Test WER: {test_results['eval_wer']:.4f}")

# 
# STEP 11: SAVE FINAL MODEL
# 
model_path = os.path.join(OUTPUT_DIR, "final_model")
processor.save_pretrained(model_path)
model.save_pretrained(model_path)
print(f"Final model saved to: {model_path}")


# DONE
print("\n" + "=" * 50)
print("PHASE 3 COMPLETE")
print("=" * 50)
print(f"Model saved to: {model_path}")
print("Next: Phase 4 - Evaluation and Error Analysis")