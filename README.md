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
  - Bitboard representation
  - Minimax algorithm
  - alpha-beta pruning
  - transposition table
  - quiescence search
  - move ordering
  - evaluation function
    - material value
    - positional value
  - book moves
  - parallelization
  - Deep Neural Network
    - Monte Carlo Tree Search - ?
    - CNN: each channel is a bitboard
    - DNN for predicting the score of a position
      - DNN for predicting the best move
      - Find large dataset of positions
      - Early pruning
      - Reinforcement learning by playing against itself
      - Q-Learning or Actor-Critic - ?

## How to run
- Clone, fork or download the project
- run 'cd Mini-Stockfish\src'
- run 'pip install pygame>=2.0.0'
- run 'python main.py'
