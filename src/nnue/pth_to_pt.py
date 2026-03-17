import torch
import torch.quantization as quant
from torch import nn
from halfkp import HalfKP_NNUE

# 1) Load the checkpoint into the HalfKP model
model = HalfKP_NNUE()
model.load_state_dict(torch.load("halfkp_best_1_33.pth", map_location="cpu"))
model.eval()

# 2) Apply dynamic INT8 quantization to Linear layers BEFORE tracing
print("Applying INT8 quantization...")
model = quant.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

# 3) Create dummy inputs matching HalfKP index shape (padded to 30)
PAD_LEN = 30
example_idx0 = torch.zeros(1, PAD_LEN, dtype=torch.long)
example_idx1 = torch.zeros(1, PAD_LEN, dtype=torch.long)

# 4) Trace the quantized model using the traceable forward() method
print("Tracing quantized model...")
traced = torch.jit.trace(model, (example_idx0, example_idx1))

# 5) Save the quantized TorchScript module
torch.jit.save(traced, "halfkp_int8.pt")
print("✓ Saved quantized INT8 TorchScript model to halfkp_int8.pt")
