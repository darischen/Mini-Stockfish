#!/usr/bin/env python3
"""
Load a trained halfkp model and evaluate it on the test set.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from pathlib import Path

# Import model classes from halfkp.py
sys.path.insert(0, str(Path(__file__).parent))
from halfkp import (
    HalfKP_NNUE, ChessDatasetHalfKP, collate_halfkp,
    HIDDEN_SIZE, MLP_HIDDEN, TABLE_SIZE, NUM_NONKING, PIECES_PER_KING,
    chunk_batch_generator
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model(model_path, dataset_path, batch_size=32768):
    """Load model and evaluate on test set."""

    print(f"Loading model from {model_path}...")
    model = HalfKP_NNUE()
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    model.eval()
    print("[OK] Model loaded")

    print(f"\nLoading dataset from {dataset_path}...")
    dataset = ChessDatasetHalfKP(dataset_path, use_cache=True)
    print(f"[OK] Dataset loaded: {len(dataset)} samples")

    # Get test indices (last 25%)
    total = len(dataset)
    test_size = total // 4
    test_indices = list(range(total - test_size, total))

    print(f"\nEvaluating on test set ({len(test_indices)} samples)...")

    # Create batch generator for test set
    gen_factory, num_batches = chunk_batch_generator(
        dataset, batch_size, subset_indices=test_indices
    )

    loss_fn = nn.SmoothL1Loss()
    total_loss = 0.0
    num_batches_processed = 0

    with torch.no_grad():
        for batch_idx, (idx0_batch, idx1_batch, targets) in enumerate(gen_factory()):
            idx0_batch = idx0_batch.to(DEVICE)
            idx1_batch = idx1_batch.to(DEVICE)
            targets = targets.to(DEVICE)

            # Forward pass
            outputs = model.forward_reset(idx0_batch, idx1_batch)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item() * len(targets)
            num_batches_processed += 1

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{num_batches}: Loss = {loss.item():.6f}")

    avg_loss = total_loss / len(test_indices)
    print(f"\n{'='*60}")
    print(f"Test Set Average Loss (SmoothL1): {avg_loss:.6f}")
    print(f"{'='*60}")

    return avg_loss

if __name__ == '__main__':
    model_path = './halfkp_best.pth'
    dataset_path = './data/combined_chessData.csv'

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found at {dataset_path}")
        sys.exit(1)

    evaluate_model(model_path, dataset_path)
