# nnue_train.py
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm  # For progress bars

# --- Helper: Forced Mate Evaluation Parser ---
def parse_evaluation(eval_str, mate_base=10000):
    """
    Parses the evaluation string from the dataset.
    - If the string starts with '#' it is interpreted as a forced mate.
      For example, "#+3" (mate in 3 moves for the side to move) is converted to mate_base - 3.
    - Otherwise, it casts the string to a float.
    
    For mate scores:
      * Mate in N moves becomes: mate_base - N
      * Mate against (indicated by a '-' sign after '#') becomes: -(mate_base - N)
      
    Adjust mate_base to set the scale of mate evaluations.
    """
    s = eval_str.strip()
    if s.startswith("#"):
        # Remove the '#' character and any whitespace
        mate_part = s[1:].strip()
        # Remove an explicit '+' if present
        mate_part = mate_part.replace("+", "")
        try:
            moves_to_mate = int(mate_part)
        except ValueError:
            moves_to_mate = 0  # Fallback if parsing fails
        # Fewer moves to mate is stronger; subtract from mate_base:
        value = mate_base - abs(moves_to_mate)
        # If the original string had a negative sign, return negative value
        if mate_part.startswith("-"):
            value = -value
        return float(value)
    else:
        return float(s)

# --- Step 1: Enhanced Feature Extraction and Accumulator Simulation ---
def enhanced_fen_to_features(fen: str) -> np.ndarray:
    """
    Convert a FEN string into an enriched feature vector.

    Base:
      - 64 squares × 12 piece‐types one-hot = 768 features

    Extras (19 total):
      1–2) White/Black king safety (adjacent friendly pawns)
      3)   Total mobility (number of legal moves)
      4–5) White/Black queen safety (attackers – defenders)
      6–7) Doubled pawns for White/Black
      8–9) Isolated pawns for White/Black
     10–11) Passed pawns for White/Black
     12–13) Hanging pieces for White/Black
     14–15) Pinned pieces for White/Black
     16–17) Knight forks for White/Black
     18–19) Trade balance for White/Black (sum of victim_value – attacker_value over legal captures)
    → final dimension = 768 + 19 = 787
    """
    board = chess.Board(fen)

    # — Base one-hot 768-vector —
    base = np.zeros(64 * 12, dtype=np.float32)
    piece_to_idx = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p:
            idx = piece_to_idx[p.piece_type] + (0 if p.color else 6)
            base[sq * 12 + idx] = 1.0

    # — Helper feature functions —
    def king_safety(color: bool) -> float:
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
        offsets = (-9, -8, -7, -1, 1, 7, 8, 9)
        safe = 0
        for o in offsets:
            n = king_sq + o
            if 0 <= n < 64 and abs((n % 8) - (king_sq % 8)) <= 1:
                npiece = board.piece_at(n)
                if npiece and npiece.color == color and npiece.piece_type == chess.PAWN:
                    safe += 1
        return float(safe)

    def queen_safety(color: bool) -> float:
        qs = list(board.pieces(chess.QUEEN, color))
        if not qs:
            return 0.0
        qsq = qs[0]
        attackers = board.attackers(not color, qsq)
        defenders = board.attackers(color, qsq)
        return float(len(attackers) - len(defenders))

    def count_doubled_pawns(color: bool) -> float:
        files = [sq % 8 for sq in board.pieces(chess.PAWN, color)]
        return float(sum(files.count(f) - 1 for f in set(files) if files.count(f) > 1))

    def count_isolated_pawns(color: bool) -> float:
        files = [sq % 8 for sq in board.pieces(chess.PAWN, color)]
        isolated = 0
        for f in files:
            if not any(
                files.count(adj) 
                for adj in (f - 1, f + 1) 
                if 0 <= adj <= 7
            ):
                isolated += 1
        return float(isolated)

    def count_passed_pawns(color: bool) -> float:
        own = board.pieces(chess.PAWN, color)
        enemy = board.pieces(chess.PAWN, not color)
        passed = 0
        for sq in own:
            f, r = sq % 8, sq // 8
            blockers = [
                e for e in enemy
                if abs((e % 8) - f) <= 1 and
                   ((e // 8) > r if color else (e // 8) < r)
            ]
            if not blockers:
                passed += 1
        return float(passed)

    def count_hanging(color: bool) -> float:
        hang = 0
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p and p.color == color:
                if board.is_attacked_by(not color, sq) and not board.is_attacked_by(color, sq):
                    hang += 1
        return float(hang)

    def count_pinned(color: bool) -> float:
        return float(sum(board.is_pinned(color, sq) for sq in chess.SQUARES))

    def count_knight_forks(color: bool) -> float:
        forks = 0
        for sq in board.pieces(chess.KNIGHT, color):
            targets = [
                t for t in board.attacks(sq)
                if board.piece_at(t) and board.piece_at(t).color != color
            ]
            if len(targets) >= 2:
                forks += 1
        return float(forks)

    # piece‐values for trade balance
    piece_value = {
        chess.PAWN:   1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK:   5,
        chess.QUEEN:  9,
        chess.KING:   0
    }

    def trade_balance(color: bool) -> float:
        # simulate legal moves for `color`
        bcopy = board.copy()
        bcopy.turn = color
        bal = 0.0
        for m in bcopy.legal_moves:
            if bcopy.is_capture(m):
                victim = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)
                if victim and attacker:
                    bal += piece_value[victim.piece_type] - piece_value[attacker.piece_type]
        return bal

    # — Compute all extras —
    w_ks   = king_safety(True)
    b_ks   = king_safety(False)
    mobility = float(len(list(board.legal_moves)))
    w_qs   = queen_safety(True)
    b_qs   = queen_safety(False)
    w_dbl  = count_doubled_pawns(True)
    b_dbl  = count_doubled_pawns(False)
    w_iso  = count_isolated_pawns(True)
    b_iso  = count_isolated_pawns(False)
    w_pas  = count_passed_pawns(True)
    b_pas  = count_passed_pawns(False)
    w_hang = count_hanging(True)
    b_hang = count_hanging(False)
    w_pin  = count_pinned(True)
    b_pin  = count_pinned(False)
    w_fork = count_knight_forks(True)
    b_fork = count_knight_forks(False)
    w_trade = trade_balance(True)
    b_trade = trade_balance(False)

    extra = np.array([
        w_ks, b_ks, mobility,
        w_qs, b_qs,
        w_dbl, b_dbl, w_iso, b_iso,
        w_pas, b_pas,
        w_hang, b_hang,
        w_pin, b_pin,
        w_fork, b_fork,
        w_trade, b_trade
    ], dtype=np.float32)

    return np.concatenate((base, extra))

# For backward compatibility, redefine fen_to_features to use the enhanced version.
def fen_to_features(fen):
    return enhanced_fen_to_features(fen)

# Simulated Accumulator class to cache the current feature vector.
class Accumulator:
    def __init__(self):
        self.features = None

    def reset(self, fen):
        """
        Compute and cache the full feature vector from the board state.
        """
        self.features = enhanced_fen_to_features(fen)
        return self.features

    def update(self, new_fen):
        """
        Simulate an incremental update:
        In a full implementation, only the changed features would be updated.
        Here we recompute the full feature vector.
        """
        new_features = enhanced_fen_to_features(new_fen)
        self.features = new_features
        return self.features

# --- Step 2: Define the NNUE Model (keeping the same depth and values) ---
class NNUEModel(nn.Module):
    def __init__(self, input_size=787, hidden_size=64):
        """
        A simple feedforward NNUE-like model.
        :param input_size: Size of the input feature vector (now 771 due to extra features)
        :param hidden_size: Number of neurons in the hidden layer.
        """
        super(NNUEModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size, 1)  # Single scalar output

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# --- Step 3: Create a Custom Dataset ---
class ChessDataset(Dataset):
    def __init__(self, csv_file):
        """
        Initialize the dataset.
        :param csv_file: Path to the CSV file containing 'FEN' and 'Evaluation' columns.
        """
        self.data = pd.read_csv(csv_file)
        self.data.columns = self.data.columns.str.strip()
        
        def is_valid_eval(val):
            try:
                _ = parse_evaluation(val)
                return True
            except Exception:
                return False
        self.data = self.data[self.data['Evaluation'].apply(is_valid_eval)]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fen = row['FEN']
        scale = 10000.0  # Use scale for normalization (adjust if needed)
        target = parse_evaluation(row['Evaluation']) / scale
        features = enhanced_fen_to_features(fen)  # Use enhanced features (771-dim)
        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor([target], dtype=torch.float32)
        return features_tensor, target_tensor

# --- Step 4: Training, Validation, and Testing the Model ---
def train_model(csv_file, num_epochs=10, batch_size=4096, learning_rate=5e-4, l2_lambda=1e-7):
    # Load the full dataset.
    full_dataset = ChessDataset(csv_file)
    total_len = len(full_dataset)
    
    # Define split ratios: 50% training, 25% validation, 25% testing
    train_len = int(0.5 * total_len)
    val_len = int(0.25 * total_len)
    test_len = total_len - train_len - val_len
    
    # Randomly split the dataset.
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = NNUEModel()  # Using our two-hidden-layer model with input size 771
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    model.train()
    accumulator = Accumulator()  # For demonstration, not used in training per se.
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0
        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1} Training")
        for i, (features, targets) in pbar:
            features = features.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            avg_loss = running_train_loss / (i + 1)
            pbar.set_postfix(train_loss=f"{avg_loss:.4e}")
        
        # Evaluate on the validation set after each epoch
        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                running_val_loss += loss.item()
        avg_val_loss = running_val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}: Average Training Loss: {avg_loss:.4e} | Validation Loss: {avg_val_loss:.4e}")
        
        # Save best model (optional)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "64indepth.pth")
        
        # Demonstrate accumulator update (optional)
        first_fen = full_dataset.data.iloc[0]['FEN']
        accumulator.reset(first_fen)
    
    print("Training complete.")
    
    # Load best model for testing (if saved)
    model.load_state_dict(torch.load("64indepth.pth"))
    model.eval()
    
    # Evaluate on the test set
    running_test_loss = 0.0
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            targets = targets.to(device)
            outputs = model(features)
            loss = criterion(outputs, targets)
            running_test_loss += loss.item()
    avg_test_loss = running_test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4e}")
    
    # Save final model (if desired)
    # torch.save(model.state_dict(), "nnue_final.pth")
    return model

if __name__ == "__main__":
    chess_df = pd.read_csv("./data/chessData.csv")
    random_df = pd.read_csv("./data/random_evals.csv")
    tactic_df = pd.read_csv("./data/tactic_evals.csv", usecols=["FEN", "Evaluation"])

    
    combined_df = pd.concat([chess_df, random_df, tactic_df], ignore_index=True)
    
    combined_csv_path = "./data/combined_chessData.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    
    csv_file = combined_csv_path  # Adjust path as needed.
    train_model(csv_file, num_epochs=10)