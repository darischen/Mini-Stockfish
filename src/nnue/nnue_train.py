# nnue_train.py
import os
import math
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
def enhanced_fen_to_features(fen):
    """
    Convert a FEN string into an enriched feature vector.
    
    This function first computes the standard 768-dimensional one-hot encoded vector
    (64 squares x 12 piece types) and then appends three extra features:
      - White king safety (count of adjacent friendly pawns)
      - Black king safety (same for black)
      - Mobility: total number of legal moves
    Final output dimension: 768 + 3 = 771.
    """
    board = chess.Board(fen)
    base_features = np.zeros(64 * 12, dtype=np.float32)
    piece_to_index = {
         chess.PAWN: 0,
         chess.KNIGHT: 1,
         chess.BISHOP: 2,
         chess.ROOK: 3,
         chess.QUEEN: 4,
         chess.KING: 5,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.piece_type]
            if not piece.color:  # chess.BLACK is False, chess.WHITE is True
                idx += 6
            base_features[square * 12 + idx] = 1.0

    # Extra Feature 1: King Safety for a given color (count adjacent friendly pawns)
    def king_safety(color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
        safety = 0.0
        offsets = [-9, -8, -7, -1, 1, 7, 8, 9]
        for offset in offsets:
            neighbor = king_sq + offset
            if neighbor < 0 or neighbor >= 64:
                continue
            # Check file difference to avoid wrap-around (difference must be at most 1)
            if abs((neighbor % 8) - (king_sq % 8)) > 1:
                continue
            n_piece = board.piece_at(neighbor)
            if n_piece is not None and n_piece.color == color and n_piece.piece_type == chess.PAWN:
                safety += 1.0
        return safety

    white_king_safety = king_safety(chess.WHITE)
    black_king_safety = king_safety(chess.BLACK)

    # Extra Feature 2: Mobility as total number of legal moves.
    mobility = float(len(list(board.legal_moves)))

    extra_features = np.array([white_king_safety, black_king_safety, mobility], dtype=np.float32)
    full_features = np.concatenate((base_features, extra_features))
    return full_features

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
    def __init__(self, input_size=771, hidden_size1=1024, hidden_size2=1024):
        """
        A simple feedforward NNUE-like model.
        :param input_size: Size of the input feature vector (now 771 due to extra features)
        :param hidden_size1: Number of neurons in the first hidden layer.
        :param hidden_size2: Number of neurons in the second hidden layer.
        """
        super(NNUEModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, 1)  # Single scalar output

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
def train_model(csv_file, num_epochs=10, batch_size=1024, learning_rate=1e-3, l2_lambda=1e-4):
    # Load the full dataset.
    full_dataset = ChessDataset(csv_file)
    total_len = len(full_dataset)
    
    # Define split ratios: 80% training, 10% validation, 10% testing
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
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
            torch.save(model.state_dict(), "best_nnue_model.pth")
        
        # Demonstrate accumulator update (optional)
        first_fen = full_dataset.data.iloc[0]['FEN']
        accumulator.reset(first_fen)
    
    print("Training complete.")
    
    # Load best model for testing (if saved)
    model.load_state_dict(torch.load("best_nnue_model.pth"))
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
    torch.save(model.state_dict(), "nnue_model_final.pth")
    return model

if __name__ == "__main__":
    chess_df = pd.read_csv("./data/chessData.csv")
    random_df = pd.read_csv("./data/random_evals.csv")
    tactic_df = pd.read_csv("./data/tactic_evals.csv", usecols=["FEN", "Evaluation"])

    
    combined_df = pd.concat([chess_df, random_df, tactic_df], ignore_index=True)
    
    combined_csv_path = "./data/combined_chessData.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    
    csv_file = combined_csv_path  # Adjust path as needed.
    train_model(csv_file, num_epochs=35)
