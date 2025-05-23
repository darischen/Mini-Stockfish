# Mini Stockfish

## Based off this tutorial
[Coding Spot's (AlejoG10) YouTube Tutorial](https://www.youtube.com/watch?v=OpL0Gcfn4B4)


[AlejoG10's GitHub Repository](https://github.com/AlejoG10/python-chess-ai-yt)

## Current improvements on the code base
- Fixed various bugs with checkmate and check detection and limiting movements of other pieces.
- Fixed erroneous checkmate detection
- Added check detection
- Added checkmate detection
- Added stalemate detection
- Added Threefold repetition detection
- Added sounds for check, checkmate, piece movement and capture, castling, promoting, and invalid moves'
- The king is highlighted in red when in check

## Future improvements
- [x] Notifying the player when a king is in check
- [x] Draw detection
  - [x] Threefold repetition
  - [x] Fifty-move rule
  - [x] Insufficient material
- AI Opponent to play against
  - [x] Bitboard representation
  - [x] Minimax algorithm
  - [x] alpha-beta pruning
  - [x] piece square tables
    - [x] Interpolation (midgame and endgame)
  - [x] transposition table
  - [x] accumulator
  - [x] quiescence search
  - [x] move ordering
  - [x] evaluation function
    - [x] material value
    - [x] positional value
  - [x] Book of opening moves
  - [x] MVV-LVA (Least Valuable Victim - Most Valuable Attacker)
  - [x] Syzygy Endgame Tablebase (Gaviota has DTM)
  - [ ] Static Exchange Evaluation
  - [x] multithreading
  - [x] Dataset of Stockfish Evaluations (16 million positions)
  - [x] Small network for fast evaluation of positions (NNUE)
  - [x] DNN for predicting the score of a position

## How to run
- Clone, fork or download the project
- run 'cd Mini-Stockfish\src'
- Download python 3.10.x
- run pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
- run 'pip install -r requirements.txt'
- run 'python main.py'