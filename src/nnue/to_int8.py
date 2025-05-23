# nnue/quantize_nnue.py
import torch
from torch import nn
# if your training code lived in nnue.nnue_train, import the same model class
from nnue_train import NNUEModel

# 1) Load your FP32 TorchScript model
fp32 = torch.jit.load("64indepth.pt", map_location="cpu").eval()

# 2) Apply dynamic quantization to every Linear layer
quantized = torch.quantization.quantize_dynamic(
    fp32,
    { nn.Linear },
    dtype=torch.qint8
)

# 3) (Optional) Freeze / optimize for inference
quantized = torch.jit.freeze(quantized)
quantized = torch.jit.optimize_for_inference(quantized)

# 4) Save out your new INT8 model
torch.jit.save(quantized, "64indepth_int8.pt")
print("Saved quantized model")
