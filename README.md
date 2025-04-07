# ChessAI

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
- Draw detection
  - [x]Threefold repetition
  - Fifty-move rule
  - Insufficient material
- Deep Learning Neural Network to play against

## How to run
- Clone, fork or download the project
- run 'cd ChessAI\src'
- run 'pip install pygame>=2.0.0'
- run 'python main.py'