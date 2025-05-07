import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
import chess

# --- Helper: Forced Mate Evaluation Parser ---
def parse_evaluation(eval_str, mate_base=10000):
    # clean BOM and whitespace
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

# --- Validation Helper ---
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
    """
    Returns two lists of indices (view=0, view=1) for a given FEN.
    """
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
                if view == 0:
                    idx0.append(idx)
                else:
                    idx1.append(idx)
    return idx0, idx1

class ChessDatasetHalfKP(Dataset):
    def __init__(self, csv_file, scale=10000.0):
        df = pd.read_csv(csv_file)
        df.columns = df.columns.str.strip()
        # remove BOM in Evaluation column
        df['Evaluation'] = df['Evaluation'].astype(str).str.lstrip('\ufeff')
        # filter only valid evaluations
        df = df[df['Evaluation'].apply(is_valid_eval)]
        self.fens = df['FEN'].tolist()
        self.targets = [parse_evaluation(v)/scale for v in df['Evaluation']]
        # precompute indices for each position
        indices = [halfkp_indices_for_fen(f) for f in self.fens]
        idx0_list, idx1_list = zip(*indices)
        # pad lists to uniform length
        self.max0 = max(len(i) for i in idx0_list)
        self.max1 = max(len(i) for i in idx1_list)
        self.idx0 = [i + [0]*(self.max0 - len(i)) for i in idx0_list]
        self.idx1 = [i + [0]*(self.max1 - len(i)) for i in idx1_list]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.idx0[idx], dtype=torch.long),
            torch.tensor(self.idx1[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )

# --- HalfKP NNUE with Batched Lookup ---
HIDDEN_SIZE = 256
MLP_HIDDEN = 32

class HalfKP_NNUE(nn.Module):
    def __init__(self, table_size=TABLE_SIZE, hidden_size=HIDDEN_SIZE, mlp_hidden=MLP_HIDDEN):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2, table_size, hidden_size) * 0.01)
        self.fc2 = nn.Linear(2*hidden_size, mlp_hidden)
        self.fc3 = nn.Linear(mlp_hidden, mlp_hidden)
        self.fc4 = nn.Linear(mlp_hidden, 1)

    def forward(self, idx0_batch, idx1_batch):
        # one-hot gather then sum
        # idx?_batch: [B, max_len]
        oh0 = F.one_hot(idx0_batch, num_classes=self.w1.size(1)).float()
        oh1 = F.one_hot(idx1_batch, num_classes=self.w1.size(1)).float()
        emb0 = torch.matmul(oh0, self.w1[0])  # [B, max0, H]
        emb1 = torch.matmul(oh1, self.w1[1])  # [B, max1, H]
        sum0 = emb0.sum(dim=1)
        sum1 = emb1.sum(dim=1)
        h = torch.cat([torch.clamp(sum0, min=0), torch.clamp(sum1, min=0)], dim=1)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4(h).squeeze(-1)

# --- Training Loop ---
def train_model(csv_file, epochs=10, batch_size=1024, lr=5e-4, l2=1e-7):
    ds = ChessDatasetHalfKP(csv_file)
    n = len(ds)
    tr, va = int(0.5*n), int(0.25*n)
    ds_tr, ds_va, ds_te = random_split(ds, [tr, va, n-tr-va])
    loader_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)
    loader_va = DataLoader(ds_va, batch_size=batch_size)
    loader_te = DataLoader(ds_te, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HalfKP_NNUE().to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.SmoothL1Loss()

    for e in range(1, epochs+1):
        model.train()
        total_loss = 0.0
        for idx0, idx1, tgt in tqdm(loader_tr, desc=f"Train Epoch {e}"):
            idx0, idx1, tgt = idx0.to(device), idx1.to(device), tgt.to(device)
            pred = model(idx0, idx1)
            loss = loss_fn(pred, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {e} Train Loss: {total_loss/len(loader_tr):.4e}")

        model.eval()
        with torch.no_grad():
            val_loss = sum(loss_fn(model(x0.to(device), x1.to(device)), y.to(device)).item()
                          for x0, x1, y in loader_va)
        print(f"Epoch {e} Val Loss: {val_loss/len(loader_va):.4e}")

    # final test
    model.eval()
    with torch.no_grad():
        test_loss = sum(loss_fn(model(x0.to(device), x1.to(device)), y.to(device)).item()
                        for x0, x1, y in loader_te)
    print(f"Test Loss: {test_loss/len(loader_te):.4e}")
    torch.save(model.state_dict(), 'halfkp_best.pth')
    return model

if __name__ == '__main__':
    # combine data files
    dfs = [pd.read_csv(p) for p in ['./data/chessData.csv','./data/random_evals.csv']]
    tact = pd.read_csv('./data/tactic_evals.csv', usecols=['FEN','Evaluation'])
    concat = pd.concat([*dfs, tact], ignore_index=True)
    concat.to_csv('./data/combined_chessData.csv', index=False)
    train_model('./data/combined_chessData.csv', epochs=20)
