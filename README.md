# Mini Stockfish

A chess engine implementation with neural network evaluation, built in Python.

## Features

### AI Engine
- Neural network position evaluation (NNUE)
- Deep neural network for score prediction
- Minimax algorithm with alpha-beta pruning
- Iterative deepening
- Quiescence search
- Transposition tables
- Move ordering (MVV-LVA)
- Opening book integration
- Syzygy endgame tablebase support
- Multithreading support
- Trained on dataset of 16 million Stockfish-evaluated positions

### Game Rules
- Complete move validation
- Check and checkmate detection
- Stalemate detection
- Draw detection (threefold repetition, fifty-move rule, insufficient material)
- Sound effects for all game events
- Visual check indicator

## Technical Implementation
- Bitboard representation
- Piece-square tables with midgame/endgame interpolation
- Custom evaluation function combining material and positional values

## How to run
- Clone, fork or download the project
- run 'cd Mini-Stockfish\src'
- Download python 3.10.x
- run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- run 'pip install -r requirements.txt'
- run 'python main.py'

## Development Notes
Initial code structure based on [tutorial](https://www.youtube.com/watch?v=OpL0Gcfn4B4), extensively modified and enhanced with AI engine, neural networks, and complete rule validation.
