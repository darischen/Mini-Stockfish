import torch
from halfkp import HalfKP_NNUE

# 1) Load the checkpoint into the HalfKP model
model = HalfKP_NNUE()
model.load_state_dict(torch.load("halfkp_best.pth", map_location="cpu"))
model.eval()

# 2) Create dummy inputs matching HalfKP index shape (padded to 30)
PAD_LEN = 30
example_idx0 = torch.zeros(1, PAD_LEN, dtype=torch.long)
example_idx1 = torch.zeros(1, PAD_LEN, dtype=torch.long)

# 3) Trace the model using the traceable forward() method
traced = torch.jit.trace(model, (example_idx0, example_idx1))

# 4) Save the TorchScript module
torch.jit.save(traced, "halfkp.pt")
print("Saved TorchScript halfkp.pt")
