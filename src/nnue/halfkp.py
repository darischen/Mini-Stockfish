import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DataLoader
import torch.optim as optim
import chess
import argparse
from tqdm import tqdm


# --- Helper: Forced Mate Evaluation Parser ---

def parse_evaluation(eval_str, mate_base=10000):
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
    def __init__(self, csv_file, scale=10000.0):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        df['Evaluation'] = df['Evaluation'].astype(str).str.lstrip('\ufeff')
        df = df[df['Evaluation'].apply(is_valid_eval)]
        self.fens = df['FEN'].tolist()
        self.targets = [parse_evaluation(v)/scale for v in df['Evaluation']]
        indices = [halfkp_indices_for_fen(f) for f in self.fens]
        idx0_list, idx1_list = zip(*indices)
        self.max0 = max(len(i) for i in idx0_list)
        self.max1 = max(len(i) for i in idx1_list)
        # pad with zeros for batch consistency
        self.idx0 = [i + [0]*(self.max0-len(i)) for i in idx0_list]
        self.idx1 = [i + [0]*(self.max1-len(i)) for i in idx1_list]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.idx0[idx], dtype=torch.long),
            torch.tensor(self.idx1[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# --- Accumulator and Sparse HalfKP NNUE Model ---
HIDDEN_SIZE = 256
MLP_HIDDEN = 32

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
        # MLP layers
        self.fc2 = nn.Linear(2 * HIDDEN_SIZE, MLP_HIDDEN)
        self.fc3 = nn.Linear(MLP_HIDDEN, MLP_HIDDEN)
        self.fc4 = nn.Linear(MLP_HIDDEN, 1)
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
        return self.fc4(x).squeeze(-1)

# Training & Export Utilities (unchanged, except adapt to new forward_reset)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train_model(csv_file,
                epochs: int = 10,
                batch_size: int = 4096,
                lr: float = 5e-4,
                l2: float = 1e-7):
    # --- prepare dataset and splits (50% train, 25% val, 25% test) ---
    ds = ChessDatasetHalfKP(csv_file)
    n = len(ds)
    n_train = int(0.50 * n)
    n_val   = int(0.25 * n)
    n_test  = n - n_train - n_val
    ds_tr, ds_va, ds_te = random_split(ds, [n_train, n_val, n_test])

    train_loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(ds_va, batch_size=batch_size)
    test_loader  = DataLoader(ds_te, batch_size=batch_size)

    # --- model, optimizer, loss ---
    model   = HalfKP_NNUE().to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, epochs + 1):
        # ——— training ———
        model.train()
        train_bar = tqdm(train_loader,
                         desc=f"Epoch {epoch}/{epochs} [Train]",
                         unit="batch")
        for x0, x1, y in train_bar:
            x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
            pred = model.forward_reset(x0, x1)
            loss = loss_fn(pred, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ——— validation ———
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader,
                           desc=f"Epoch {epoch}/{epochs} [Val]  ",
                           unit="batch")
            for x0, x1, y in val_bar:
                x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
                pred = model.forward_reset(x0, x1)
                batch_loss = loss_fn(pred, y).item()
                val_loss += batch_loss * y.size(0)
                val_bar.set_postfix({"val_loss": f"{batch_loss:.4f}"})
        val_loss /= len(ds_va)
        print(f"→ Epoch {epoch}: Validation Loss = {val_loss:.4f}\n")

    # ——— final testing ———
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Test                ", unit="batch")
        for x0, x1, y in test_bar:
            x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
            pred = model.forward_reset(x0, x1)
            batch_loss = loss_fn(pred, y).item()
            test_loss += batch_loss * y.size(0)
            test_bar.set_postfix({"test_loss": f"{batch_loss:.4f}"})
    test_loss /= len(ds_te)
    print(f"→ Final Test Loss = {test_loss:.4f}")

    # Save the trained weights
    torch.save(model.state_dict(), 'halfkp_accum.pth')
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train'], default='train')
    parser.add_argument('--data', default='./data/combined_chessData.csv')
    args = parser.parse_args()
    if args.mode == 'train':
        train_model(args.data)
