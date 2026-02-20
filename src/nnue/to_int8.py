# nnue/to_int8.py
import torch
from torch import nn

# 1) Load the FP32 TorchScript HalfKP model
fp32 = torch.jit.load("halfkp.pt", map_location="cpu").eval()

# 2) Apply dynamic quantization to every Linear layer
quantized = torch.quantization.quantize_dynamic(
    fp32,
    { nn.Linear },
    dtype=torch.qint8
)

# 3) Freeze and optimize for inference
quantized = torch.jit.freeze(quantized)
quantized = torch.jit.optimize_for_inference(quantized)

# 4) Save the INT8 model
torch.jit.save(quantized, "halfkp_int8.pt")
print("Saved quantized halfkp_int8.pt")
