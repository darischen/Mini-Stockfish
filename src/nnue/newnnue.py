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
import optuna  # For hyperparameter search

def parse_evaluation(eval_str, mate_base=10000):
    s = eval_str.strip()
    if s.startswith("#"):
        mate_part = s[1:].strip()
        mate_part = mate_part.replace("+", "")
        try:
            moves_to_mate = int(mate_part)
        except ValueError:
            moves_to_mate = 0
        value = mate_base - abs(moves_to_mate)
        if mate_part.startswith("-"):
            value = -value
        return float(value)
    else:
        return float(s)

def enhanced_fen_to_features(fen):
    """
    Convert a FEN string into an enriched 771-dimensional feature vector.
    The first 768 dimensions represent a one-hot encoding of piece placements 
    (64 squares x 12 piece types) and the last 3 are extra features (e.g., king safety,
    mobility).
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
            if abs((neighbor % 8) - (king_sq % 8)) > 1:
                continue
            n_piece = board.piece_at(neighbor)
            if n_piece is not None and n_piece.color == color and n_piece.piece_type == chess.PAWN:
                safety += 1.0
        return safety

    white_king_safety = king_safety(chess.WHITE)
    black_king_safety = king_safety(chess.BLACK)
    mobility = float(len(list(board.legal_moves)))
    
    extra_features = np.array([white_king_safety, black_king_safety, mobility], dtype=np.float32)
    full_features = np.concatenate((base_features, extra_features))
    return full_features

def fen_to_features(fen):
    return enhanced_fen_to_features(fen)

# --- Revised NNUE Model Architecture ---
class NNUEModel(nn.Module):
    def __init__(self, hidden_size=1024):
        """
        This NNUE-like model mimics Stockfish’s efficient architecture.
        It splits the input features into two parts:
          - The first 768 dimensions (one-hot board features) are processed via an embedding (lookup) layer.
          - The remaining 3 extra features (e.g., king safety and mobility) are processed via a small FC layer.
        Their contributions are summed (with a learned bias), passed through ReLU,
        and then mapped to a single scalar evaluation.
        
        :param hidden_size: Number of neurons in the hidden layer.
        """
        super(NNUEModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer for board features.
        # This simulates the accumulation of weights for active (nonzero) board features.
        self.embeddings = nn.Embedding(768, hidden_size)
        
        # Learned bias for the accumulated board feature representation.
        self.hidden_bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Fully connected layer for the 3 extra features.
        self.extra_fc = nn.Linear(3, hidden_size)
        
        # Final layer mapping the combined hidden representation to a single scalar evaluation.
        self.fc_out = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 771) where:
           - x[:, :768] are the sparse board features (typically one-hot encoded).
           - x[:, 768:] are the 3 extra features.
        """
        board_features = x[:, :768]   # [batch_size, 768]
        extra_features = x[:, 768:]   # [batch_size, 3]
        
        # For the board features, multiply by the embedding weight matrix.
        # With a one-hot encoding, this is equivalent to summing the weights
        # for the active features.
        accumulated_board = torch.matmul(board_features, self.embeddings.weight)  # Shape: [batch_size, hidden_size]
        accumulated_board = accumulated_board + self.hidden_bias
        
        # Process the extra features through the fully connected layer.
        extra_transformed = self.extra_fc(extra_features)  # Shape: [batch_size, hidden_size]
        
        # Combine both contributions.
        hidden = accumulated_board + extra_transformed
        
        # Apply ReLU activation.
        hidden = torch.relu(hidden)
        
        # Final output layer produces the scalar evaluation.
        output = self.fc_out(hidden)
        return output

# --- Custom Dataset Definition ---
class ChessDataset(Dataset):
    def __init__(self, csv_file):
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
        scale = 10000.0  # Normalization scale
        target = parse_evaluation(row['Evaluation']) / scale
        features = enhanced_fen_to_features(fen)
        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor([target], dtype=torch.float32)
        return features_tensor, target_tensor

# --- Training Function ---
def train_model(csv_file, num_epochs=10, batch_size=8192, learning_rate=1e-3, l2_lambda=1e-7, hidden_size=1024):
    full_dataset = ChessDataset(csv_file)
    total_len = len(full_dataset)
    
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    test_len = total_len - train_len - val_len
    
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = NNUEModel(hidden_size=hidden_size)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    model.train()
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "newbest.pth")
    
    print("Training complete.")
    model.load_state_dict(torch.load("newbest.pth"))
    model.eval()
    
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
    # torch.save(model.state_dict(), "newfinal.pth")
    return model

# --- Optuna Objective Function ---
def objective(trial):
    # Define hyperparameter search space.
    learning_rate = trial.suggest_float("learning_rate", 1e-7, 1e1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [512, 2048, 4096, 8192, 16384])
    hidden_size = trial.suggest_categorical("hidden_size", [512, 771, 1024, 1542, 2048])
    num_epochs = 1
    l2_lambda = trial.suggest_float("l2_lambda", 1e-9, 1e-3, log=True)
    
    # Use the global CSV file (set later in __main__) as the dataset.
    full_dataset = ChessDataset(csv_file)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = int(0.1 * total_len)
    _ = total_len - train_len - val_len  # Unused test split for hyperparameter optimization
    
    train_dataset, val_dataset, _ = random_split(full_dataset, [train_len, val_len, total_len - train_len - val_len])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = NNUEModel(hidden_size=hidden_size)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # Validation step.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)
                outputs = model(features)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        best_val_loss = min(best_val_loss, avg_val_loss)
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return best_val_loss

if __name__ == "__main__":
    # Prepare the combined CSV file.
    chess_df = pd.read_csv("./data/chessData.csv")
    random_df = pd.read_csv("./data/random_evals.csv")
    tactic_df = pd.read_csv("./data/tactic_evals.csv", usecols=["FEN", "Evaluation"])
    combined_df = pd.concat([chess_df, random_df, tactic_df], ignore_index=True)
    combined_csv_path = "./data/combined_chessData.csv"
    combined_df.to_csv(combined_csv_path, index=False)
    csv_file = combined_csv_path  # This global variable is used in both train_model and objective.
    
    # Create a tqdm progress bar for the hyperparameter search.
    n_trials = 10  # Adjust the number of trials as needed.
    pbar = tqdm(total=n_trials, desc="Hyperparameter Search")
    
    # Define a callback to update the progress bar.
    def tqdm_callback(study, trial):
        pbar.update(1)
        pbar.set_postfix(best=study.best_value)
    
    # Run hyperparameter search with Optuna.
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, callbacks=[tqdm_callback])
    
    pbar.close()
    
    print("Best trial:")
    best_trial = study.best_trial
    print(f"  Value: {best_trial.value}")
    print("  Hyperparameters: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Optionally, you can now run a full training using the best hyperparameters:
    # model = train_model(csv_file,
    #                     num_epochs=50,
    #                     batch_size=best_trial.params['batch_size'],
    #                     learning_rate=best_trial.params['learning_rate'],
    #                     l2_lambda=best_trial.params['l2_lambda'],
    #                     hidden_size=best_trial.params['hidden_size'])
