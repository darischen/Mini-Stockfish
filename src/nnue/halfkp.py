import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
import chess
import argparse
from tqdm import tqdm

print(f'GPU available: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')

# --- Helper: Forced Mate Evaluation Parser ---

def parse_evaluation(eval_str, mate_base=3000):
    """
    Parse evaluation string with capped mate scores.
    mate_base=3000 ensures mate scores don't dominate training.
    """
    s = str(eval_str).lstrip('\ufeff').strip()
    if s.startswith("#"):
        mate_part = s[1:].replace("+", "").strip()
        try:
            moves_to_mate = int(mate_part)
        except ValueError:
            moves_to_mate = 0
        val = mate_base - abs(moves_to_mate)
        if mate_part.startswith("-"):
            val = -val
        return float(val)
    try:
        return float(s)
    except ValueError:
        raise ValueError(f"Invalid evaluation string: '{eval_str}'")


def is_valid_eval(eval_str):
    try:
        parse_evaluation(eval_str)
        return True
    except:
        return False

# --- Dataset with Precomputed HalfKP Indices ---
NUM_NONKING = 10
TABLE_SIZE = 64 * NUM_NONKING

piece_to_idx = {
    (True,  chess.PAWN):   0,
    (True,  chess.KNIGHT): 1,
    (True,  chess.BISHOP): 2,
    (True,  chess.ROOK):   3,
    (True,  chess.QUEEN):  4,
    (False, chess.PAWN):   5,
    (False, chess.KNIGHT): 6,
    (False, chess.BISHOP): 7,
    (False, chess.ROOK):   8,
    (False, chess.QUEEN):  9,
}


def halfkp_indices_for_fen(fen):
    board = chess.Board(fen)
    idx0, idx1 = [], []
    for view in (0,1):
        king_sq = board.king(bool(view))
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece and piece.piece_type != chess.KING:
                base_idx = king_sq * NUM_NONKING
                offset = piece_to_idx[(piece.color, piece.piece_type)]
                idx = base_idx + offset
                (idx0 if view==0 else idx1).append(idx)
    return idx0, idx1

class ChessDatasetHalfKP(Dataset):
    def __init__(self, csv_file, scale=400.0):
        print(f"Loading CSV from {csv_file}...")
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded {len(df)} rows")

        print("Cleaning column names...")
        df.columns = df.columns.str.strip()
        print("✓ Columns cleaned")

        print("Processing evaluations...")
        df['Evaluation'] = df['Evaluation'].astype(str).str.lstrip('\ufeff')
        df = df[df['Evaluation'].apply(is_valid_eval)]
        print(f"✓ Filtered to {len(df)} valid evaluations")

        print("Extracting FENs and targets...")
        self.fens = df['FEN'].tolist()
        self.targets = [parse_evaluation(v)/scale for v in tqdm(df['Evaluation'], desc="Parsing evaluations", unit="eval")]
        print(f"✓ Extracted {len(self.fens)} FENs")

        print("Computing HalfKP indices...")
        indices = [halfkp_indices_for_fen(f) for f in tqdm(self.fens, desc="Computing HalfKP indices", unit="fen")]
        print(f"✓ Computed indices for {len(indices)} positions")

        idx0_list, idx1_list = zip(*indices)
        self.max0 = max(len(i) for i in idx0_list)
        self.max1 = max(len(i) for i in idx1_list)
        print(f"  Max view0 features: {self.max0}, Max view1 features: {self.max1}")

        # pad with zeros for batch consistency
        print("Padding indices for batch consistency...")
        self.idx0 = [i + [0]*(self.max0-len(i)) for i in tqdm(idx0_list, desc="Padding view0 indices", unit="idx")]
        self.idx1 = [i + [0]*(self.max1-len(i)) for i in tqdm(idx1_list, desc="Padding view1 indices", unit="idx")]
        print(f"✓ Dataset ready with {len(self.fens)} samples")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.idx0[idx], dtype=torch.long),
            torch.tensor(self.idx1[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# --- Accumulator and Sparse HalfKP NNUE Model ---
HIDDEN_SIZE = 1024  # Increased from 256 (4x)
MLP_HIDDEN = 256    # Increased from 32 (8x)

class SparseAccumulator:
    """
    Maintains a sparse accumulator state for HalfKP features.
    """
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.acc0 = None
        self.acc1 = None

    def reset(self, idx0, idx1, emb0, emb1):
        # idx0, idx1: LongTensors of shape (batch, L0) and (batch, L1)
        sum0 = emb0(idx0).sum(dim=1)      # (batch, H)
        sum1 = emb1(idx1).sum(dim=1)
        self.acc0 = torch.clamp(sum0, min=0)
        self.acc1 = torch.clamp(sum1, min=0)
        return torch.cat([self.acc0, self.acc1], dim=1)

    def update(self, removed0, added0, removed1, added1, emb0, emb1):
        # removed*/added*: indices removed/added since last state
        delta0 = emb0(added0).sum(dim=1) - emb0(removed0).sum(dim=1)
        delta1 = emb1(added1).sum(dim=1) - emb1(removed1).sum(dim=1)
        self.acc0 = torch.clamp(self.acc0 + delta0, min=0)
        self.acc1 = torch.clamp(self.acc1 + delta1, min=0)
        return torch.cat([self.acc0, self.acc1], dim=1)

class HalfKP_NNUE(nn.Module):
    def __init__(self, max0=0, max1=0):
        super().__init__()
        # Embedding tables instead of dense one-hot
        self.emb0 = nn.Embedding(TABLE_SIZE, HIDDEN_SIZE)
        self.emb1 = nn.Embedding(TABLE_SIZE, HIDDEN_SIZE)
        # Initialize embeddings small
        nn.init.normal_(self.emb0.weight, std=0.01)
        nn.init.normal_(self.emb1.weight, std=0.01)
        # Deeper MLP network with more capacity
        input_size = 2 * HIDDEN_SIZE  # 2048
        self.fc2 = nn.Linear(input_size, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, MLP_HIDDEN)  # 256
        self.fc5 = nn.Linear(MLP_HIDDEN, 64)
        self.fc_out = nn.Linear(64, 1)
        # Sparse accumulator instance (not a parameter)
        self.accumulator = SparseAccumulator(HIDDEN_SIZE)

    def forward_reset(self, idx0_batch, idx1_batch):
        # initial full reset
        h = self.accumulator.reset(idx0_batch, idx1_batch, self.emb0, self.emb1)
        return self._mlp(h)

    def forward_update(self, rem0, add0, rem1, add1):
        # incremental update: indices of removed/added per view
        h = self.accumulator.update(rem0, add0, rem1, add1, self.emb0, self.emb1)
        return self._mlp(h)

    def _mlp(self, h):
        x = F.relu(self.fc2(h))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return self.fc_out(x).squeeze(-1)

# Training & Export Utilities (unchanged, except adapt to new forward_reset)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(csv_file,
                epochs: int = 50,
                batch_size: int = 8192,
                lr: float = 2e-4,
                l2: float = 1e-7):
    print("=" * 70)
    print("TRAINING HALFKP NNUE MODEL")
    print("=" * 70)

    print("\n[1/6] Loading dataset...")
    ds = ChessDatasetHalfKP(csv_file)
    n = len(ds)
    n_train = int(0.50 * n)
    n_val   = int(0.25 * n)
    n_test  = n - n_train - n_val
    print(f"✓ Dataset loaded: {n} total samples")

    print(f"\n[2/6] Splitting data...")
    ds_tr, ds_va, ds_te = random_split(ds, [n_train, n_val, n_test])
    print(f"  • Training:   {n_train} samples ({100*n_train/n:.1f}%)")
    print(f"  • Validation: {n_val} samples ({100*n_val/n:.1f}%)")
    print(f"  • Test:       {n_test} samples ({100*n_test/n:.1f}%)")

    print(f"\n[3/6] Creating data loaders (batch_size={batch_size})...")
    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ds_va, batch_size=batch_size)
    test_loader  = DataLoader(ds_te, batch_size=batch_size)
    print(f"✓ Data loaders created")
    print(f"  • Training batches:   {len(train_loader)}")
    print(f"  • Validation batches: {len(val_loader)}")
    print(f"  • Test batches:       {len(test_loader)}")

    print(f"\n[4/6] Initializing model on {DEVICE}...")
    model   = HalfKP_NNUE().to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.SmoothL1Loss()
    print(f"✓ Model initialized")
    print(f"  • Device: {DEVICE}")
    print(f"  • Learning rate: {lr}")
    print(f"  • L2 regularization: {l2}")

    best_val_loss = float('inf')
    best_model_path = 'halfkp_best.pth'

    print(f"\n[5/6] Starting training for {epochs} epochs...")
    print("=" * 70)

    for epoch in range(1, epochs + 1):
        # ——— training ———
        model.train()
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch}/{epochs} [TRAIN]",
                         unit="batch",
                         ncols=100)
        train_loss_total = 0.0
        for x0, x1, y in train_bar:
            x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
            pred = model.forward_reset(x0, x1)
            loss = loss_fn(pred, y)
            train_loss_total += loss.item() * y.size(0)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss_avg = train_loss_total / n_train
        print(f"  ✓ Training Loss (avg): {train_loss_avg:.4f}")

        # ——— validation ———
        print(f"  → Running validation on {len(val_loader)} batches...")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x0, x1, y in tqdm(val_loader, desc="Epoch {}/{} [VALID]".format(epoch, epochs), unit="batch", ncols=100, leave=False):
                x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
                pred = model.forward_reset(x0, x1)
                batch_loss = loss_fn(pred, y).item()
                val_loss += batch_loss * y.size(0)
        val_loss /= len(ds_va)
        print(f"  ✓ Validation Loss: {val_loss:.4f}")

        # <-- save best model if improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"  *** 🌟 BEST MODEL SAVED to {best_model_path} (Val Loss: {val_loss:.4f}) ***")
        else:
            print(f"  (best so far: {best_val_loss:.4f})")
        print()

    # ——— final testing ———
    print("=" * 70)
    print(f"[6/6] Final evaluation on test set...")
    print(f"  → Running test on {len(test_loader)} batches...")
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x0, x1, y in tqdm(test_loader, desc="[TEST]", unit="batch", ncols=100):
            x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
            pred = model.forward_reset(x0, x1)
            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss * y.size(0)
    test_loss /= len(ds_te)
    print(f"✓ Test Loss: {test_loss:.4f}")
    print("=" * 70)

    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--data', default='./data/combined_chessData.csv')
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args.data)
