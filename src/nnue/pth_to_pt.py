import torch
from nnue_train import NNUEModel

# 1) Load the checkpoint into the PyTorch model
model = NNUEModel(input_size=771)
model.load_state_dict(torch.load("best_nnue.pth", map_location="cpu"))
model.eval()

# 2) Create a dummy input matching your feature shape
example = torch.zeros(1, 771)

# 3) Trace (or script) the model
traced = torch.jit.trace(model, example)
# If you have any control flow in forward, use script:
# traced = torch.jit.script(model)

# 4) Save the TorchScript module
torch.jit.save(traced, "best_nnue.pt")
print("Saved TorchScript to best_nnue.pt")
