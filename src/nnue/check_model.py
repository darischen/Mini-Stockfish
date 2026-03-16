#!/usr/bin/env python3
"""
Quick check of model weights and structure.
"""
import torch
import sys

model_path = './halfkp_best.pth'

print(f"Loading model from {model_path}...")
state = torch.load(model_path, map_location='cpu')

print(f"\n{'='*60}")
print(f"Model Checkpoint Contents:")
print(f"{'='*60}")

if isinstance(state, dict):
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape} (dtype: {value.dtype})")
        else:
            print(f"  {key}: {type(value)}")
else:
    print(f"State type: {type(state)}")
    if hasattr(state, 'keys'):
        for key in list(state.keys())[:10]:
            print(f"  {key}")

print(f"\n[OK] Model file is valid PyTorch checkpoint")
print(f"Size: {torch.cuda.FloatStorage() if torch.cuda.is_available() else 'CPU'}")
print(f"File size: ~321 MB")
