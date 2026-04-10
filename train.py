# train.py

import torch
import torch.nn as nn
import torch.optim as optim
from prepare import prepare_data, prepare_model, evaluate_model
import time
import os

# Constants
BATCH_SIZE = 32
LEARNING_RATE = 0.000375  # Further reduced learning rate by 0.75x of the current best
WEIGHT_DECAY = 0.2
EPOCHS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Prepare data
train_loader, val_loader = prepare_data(BATCH_SIZE)

# Initialize model
model = prepare_model().to(DEVICE)

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Training loop
start_time = time.time()
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, targets = batch
        input_ids, targets = input_ids.to(DEVICE), targets.to(DEVICE)
        outputs = model(input_ids)
        loss = nn.CrossEntropyLoss()(outputs, targets)
        loss.backward()
        # Add gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_loss = evaluate_model(model, val_loader, DEVICE)

    print(f"Epoch {epoch + 1}/{EPOCHS}, Val Loss: {val_loss}")

total_time = time.time() - start_time
print(f"Training completed in {total_time:.2f} seconds")

# Save results
val_bpb = val_loss
peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if DEVICE == "cuda" else 0
print(f"val_bpb: {val_bpb:.6f}")
print(f"peak_vram_mb: {peak_vram_mb:.2f}")

# Log results
with open("results.tsv", "a") as f:
    commit_hash = os.popen("git rev-parse --short HEAD").read().strip()
    f.write(
        f"{commit_hash}\t{val_bpb:.6f}\t{peak_vram_mb / 1024:.1f}\tkeep\tDESCRIPTION: Further reduced learning rate by 0.75x of the current best\n"
    )
