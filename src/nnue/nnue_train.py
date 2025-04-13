# nnue_train.py
import os
import math
import pandas as pd
import numpy as np
import chess
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm  # Import tqdm for progress bars

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

# --- Step 1: Convert FEN to Feature Vector ---
def fen_to_features(fen):
    """
    Convert a FEN string into a 768-dimensional one-hot encoded feature vector.
    
    We assume the following ordering for piece types:
      Indices 0-5: White Pawn, Knight, Bishop, Rook, Queen, King
      Indices 6-11: Black Pawn, Knight, Bishop, Rook, Queen, King
    For each of the 64 squares, we create a binary vector of length 12.
    The final vector is flattened into a 768-dimensional numpy array.
    """
    board = chess.Board(fen)
    features = np.zeros(64 * 12, dtype=np.float32)
    
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
            features[square * 12 + idx] = 1.0
    return features

# --- Step 2: Define the NNUE Model ---
class NNUEModel(nn.Module):
    def __init__(self, input_size=768, hidden_size1=1024, hidden_size2=512):
        """
        A simple feedforward NNUE-like model.
        :param input_size: Size of the input feature vector (768 = 64 * 12)
        :param hidden_size: Number of neurons in the hidden layer.
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
        # Strip whitespace from column names in case there are leading/trailing spaces
        self.data.columns = self.data.columns.str.strip()
        
        # Filter out rows where Evaluation cannot be parsed (if desired)
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
        scale = 10000.0
        target = parse_evaluation(row['Evaluation']) / scale
        features = fen_to_features(fen)
        features_tensor = torch.from_numpy(features)
        target_tensor = torch.tensor([target], dtype=torch.float32)
        return features_tensor, target_tensor

# --- Step 4: Training the Model ---
def train_model(csv_file, num_epochs=10, batch_size=1024, learning_rate=1e-3, l2_lambda=1e-7):
    # Create the dataset and dataloader.
    dataset = ChessDataset(csv_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the NNUE model.
    model = NNUEModel()
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        # Wrap the dataloader in tqdm for a progress bar
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}")
        for i, (features, targets) in pbar:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            # Update progress bar with current average loss
            avg_loss = running_loss / (i + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4e}")
    
    print("Training complete.")
    torch.save(model.state_dict(), "nnue_model.pth")
    return model

if __name__ == "__main__":
    # Change the csv file path if needed.
    csv_file = "chessData.csv"
    train_model(csv_file, num_epochs=15)
