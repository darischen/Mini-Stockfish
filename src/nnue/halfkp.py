import os
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, random_split, DataLoader
from tqdm import tqdm
import chess
import argparse

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
        self.idx0 = [i + [0]*(self.max0-len(i)) for i in idx0_list]
        self.idx1 = [i + [0]*(self.max1-len(i)) for i in idx1_list]
    def __len__(self): return len(self.targets)
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
    def __init__(self):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(2, TABLE_SIZE, HIDDEN_SIZE) * 0.01)
        self.fc2 = nn.Linear(2*HIDDEN_SIZE, MLP_HIDDEN)
        self.fc3 = nn.Linear(MLP_HIDDEN, MLP_HIDDEN)
        self.fc4 = nn.Linear(MLP_HIDDEN, 1)
    def forward(self, idx0_batch, idx1_batch):
        oh0 = F.one_hot(idx0_batch, num_classes=self.w1.size(1)).float()
        oh1 = F.one_hot(idx1_batch, num_classes=self.w1.size(1)).float()
        emb0 = torch.matmul(oh0, self.w1[0])
        emb1 = torch.matmul(oh1, self.w1[1])
        sum0 = emb0.sum(dim=1)
        sum1 = emb1.sum(dim=1)
        h = torch.cat([torch.clamp(sum0,0), torch.clamp(sum1,0)], dim=1)
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        return self.fc4(h).squeeze(-1)

# --- Training & Export Utilities ---

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os

# Training function remains unchanged

def train_model(csv_file, epochs=10, batch_size=4096, lr=5e-4, l2=1e-7):
    ds = ChessDatasetHalfKP(csv_file)
    n = len(ds)
    tr, va = int(0.5*n), int(0.25*n)
    ds_tr, ds_va, ds_te = random_split(ds, [tr, va, n-tr-va])
    loader = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(ds_va, batch_size=batch_size, num_workers=8)
    test_loader = DataLoader(ds_te, batch_size=batch_size, num_workers=8)
    model = HalfKP_NNUE().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=l2)
    loss_fn = nn.SmoothL1Loss()
    for e in range(1, epochs+1):
        model.train()
        tl = 0.0
        for x0, x1, y in tqdm(loader, desc=f"Epoch {e}"):
            x0, x1, y = x0.to(DEVICE), x1.to(DEVICE), y.to(DEVICE)
            pred = model(x0, x1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item()
        print(f"Train Loss {tl/len(loader):.4e}")
        model.eval()
        vl = sum(loss_fn(model(x0.to(DEVICE), x1.to(DEVICE)), y.to(DEVICE)).item()
                 for x0, x1, y in val_loader)
        print(f"Val Loss {vl/len(val_loader):.4e}")
    model.eval()
    tloss = sum(loss_fn(model(x0.to(DEVICE), x1.to(DEVICE)), y.to(DEVICE)).item()
                for x0, x1, y in test_loader)
    print(f"Test Loss {tloss/len(test_loader):.4e}")
    torch.save(model.state_dict(), 'halfkp_best.pth')
    return model

# Export utility now only uses passed max0,max1

def export_torchscript(model, max0, max1):
    print(f"[export] called with max0={max0}, max1={max1}")
    # trace raw two-input model
    dummy0 = torch.zeros(1, max0, dtype=torch.long, device=DEVICE)
    dummy1 = torch.zeros(1, max1, dtype=torch.long, device=DEVICE)
    traced_halfkp = torch.jit.trace(model, (dummy0, dummy1))
    print("[export] raw HalfKP traced")

    # wrap into single-input
    class SingleInputWrapper(torch.nn.Module):
        def __init__(self, halfkp_module, split_idx):
            super().__init__()
            self.halfkp = halfkp_module
            self.split = split_idx
        def forward(self, x):
            idx0 = x[:, :self.split]
            idx1 = x[:, self.split:]
            return self.halfkp(idx0, idx1)
    wrapper = SingleInputWrapper(traced_halfkp, max0)
    print("[export] wrapper created")

    # trace & save single-input
    dummy_concat = torch.zeros(1, max0 + max1, dtype=torch.long, device=DEVICE)
    traced_wrapper = torch.jit.trace(wrapper, dummy_concat)
    torch.jit.save(traced_wrapper, "halfkp.pt")
    print("[export] Saved single-input TorchScript as halfkp.pt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train','export'], default='train')
    parser.add_argument('--data', default='./data/combined_chessData.csv')
    parser.add_argument('--max0', type=int, help='padding length idx0 for export')
    parser.add_argument('--max1', type=int, help='padding length idx1 for export')
    args = parser.parse_args()
    print(f"[main] mode={args.mode}, data={args.data}, max0={args.max0}, max1={args.max1}")
    ds = ChessDatasetHalfKP('./data/combined_chessData.csv')
    print("max0 =", ds.max0)
    print("max1 =", ds.max1)

    if args.mode == 'train':
        train_model(args.data, epochs=25)
    else:
        if args.max0 is None or args.max1 is None:
            raise ValueError("Export mode requires --max0 and --max1 flags")
        print("[main] Loading model for export")
        model = HalfKP_NNUE().to(DEVICE)
        model.load_state_dict(torch.load('halfkp_best.pth', map_location=DEVICE))
        model.eval()
        export_torchscript(model, args.max0, args.max1)
        print("[main] export complete")
