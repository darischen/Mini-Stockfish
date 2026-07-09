# Mini-Stockfish Chess Engine

A feature-rich chess engine with neural network (NNUE) evaluation, written in Python with performance-critical components in Cython. Includes a complete pygame-based GUI with move history navigation, pawn promotion, and support for endgame tablebases.

## Features

### Core Chess Engine
- **Search Algorithm:** Minimax with alpha-beta pruning, iterative deepening, and quiescence search
- **Neural Network Evaluation:** HalfKP-based NNUE (40,960 features) trained on Stockfish positions
- **Move Ordering:** MVV-LVA, killer moves, and counter-move heuristics
- **Transposition Tables:** Position caching to avoid redundant search
- **Opening Book:** Integration with chess opening book (20-ply depth)
- **Endgame Support:** Syzygy and Gaviota tablebase support
- **Performance:** Optimized search in Cython (core_search.pyx)

### Game Rules & Validation
- **Complete Chess Rules:** Check, checkmate, stalemate detection
- **Draw Detection:** Threefold repetition, fifty-move rule, insufficient material
- **Pawn Promotion:** Interactive modal dialog for promotion piece selection
- **En Passant & Castling:** Full support with validation
- **FEN Support:** Load positions via FEN notation
- **Move History:** Undo/redo with full game state tracking

### User Interface
- **Pygame GUI:** Drag-and-drop piece movement with visual feedback
- **Customizable Themes:** Multiple color schemes for board and pieces
- **Board Perspective:** Flip board between white and black viewpoints
- **Move Visualization:** Last move highlighting, legal move indicators, check indicator
- **Move History Navigation:** Step through previous moves with arrow keys
- **Sound Effects:** Audio feedback for moves, captures, and game events
- **Promotion Modal:** Clean UI for selecting promotion piece

### AI Capabilities
- **Configurable Depth:** Default depth 10 plies (adjustable)
- **NNUE Evaluation:** 8-bit quantized neural network for fast position scoring
- **Dual Evaluation:** Combine material values with neural network scores
- **Multithreading Ready:** Infrastructure for parallel search (future enhancement)
- **Model Support:** TorchScript models for efficient inference

## Architecture Overview

```
┌─────────────────────────────────────┐
│         UI Layer (Pygame)            │
│  main.py, game.py, dragger.py,      │
│  theme.py, config.py, promotion.py  │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│     Game Logic Layer                 │
│  board.py, piece.py, move.py,       │
│  square.py, color.py, sound.py      │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│    Search Engine Layer (Cython)      │
│  ai.py, core_search.pyx,            │
│  bitboard.py, accumulator.py        │
└─────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────┐
│  Neural Network Layer                │
│  halfkp.py, nnue_train.py,          │
│  evaluate_model.py, check_model.py  │
└─────────────────────────────────────┘
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Application entry point, event loop, pygame window |
| `game.py` | Game state management, rendering, move history |
| `board.py` | Board representation, move validation, rule enforcement |
| `ai.py` | AI decision-making, book lookups, promotion selection |
| `core_search.pyx` | High-performance minimax with transposition tables (Cython) |
| `accumulator.py` | Incremental NNUE feature computation for efficiency |
| `halfkp.py` | HalfKP encoding, dataset processing for training |
| `bitboard.py` | Bitboard operations for move generation |

## Setup & Installation

### Requirements
- Python 3.10+
- CUDA 11.8 (for GPU acceleration, optional)

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Mini-Stockfish.git
   cd Mini-Stockfish
   ```

2. **Set up Python environment** (recommended)
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install PyTorch** (with CUDA 11.8 support)
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install dependencies**
   ```bash
   cd src
   pip install -r requirements.txt
   ```

5. **Build Cython extension** (for optimized search)
   ```bash
   python setup.py build_ext --inplace
   ```

6. **Run the game**
   ```bash
   python main.py
   ```

## How to Play

### Basic Controls
| Action | Key/Input |
|--------|-----------|
| Move piece | Drag piece to destination |
| AI move (Black) | Press `A` |
| AI move (White) | Press `S` |
| Load FEN | Press `E` (enter FEN at prompt) |
| Reset game | Press `R` |
| Flip board | Press `F` |
| Change theme | Press `T` |
| Undo move | Press `←` (left arrow) |
| Redo move | Press `→` (right arrow) |
| Jump to start | Press `↓` (down arrow) |
| Jump to end | Press `↑` (up arrow) |

### AI Settings
Edit the AI configuration in `main.py`:
```python
self.ai = ChessAI(depth=10, use_dnn=True)
```
- `depth`: Search depth in plies (default: 10, range: 2-20)
- `use_dnn`: Enable/disable NNUE evaluation (default: True)

## Technical Details

### Search Strategy
1. **Iterative Deepening:** Gradually increase search depth for time management
2. **Alpha-Beta Pruning:** Eliminate branches that can't affect the result
3. **Transposition Tables:** Cache evaluated positions to avoid redundant work
4. **Quiescence Search:** Extend search in tactical positions (captures, checks)
5. **Move Ordering:**
   - Killer moves (same move works well across similar positions)
   - Counter-move heuristic (responses to opponent moves)
   - MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
   - Hash moves from transposition table

### NNUE Architecture
- **Input Layer:** 40,960 HalfKP features (piece-square combinations relative to kings)
- **Hidden Layers:** ReLU-activated fully connected layers
- **Output:** Single evaluation neuron (capped mate scores at 3000)
- **Quantization:** 8-bit integer weights for efficient inference
- **Accumulator:** Incremental computation tracks feature changes across search

### Performance Optimizations
- **Cython Core Search:** Hot path (minimax loop) in compiled Cython
- **Numba JIT:** Accumulator updates and neural network inference
- **Bitboard Representation:** Fast move generation and board state
- **TorchScript Model:** Pre-compiled model for CPU/GPU inference
- **Incremental Updates:** Avoid full feature recomputation for each board state

### Board Synchronization
- **Internal Bitboard:** Custom Python bitboard for move validation
- **Python-Chess Integration:** Synchronized board for:
  - Standard Algebraic Notation (SAN) conversion
  - UCI move format
  - FEN support
  - Tablebase lookups

## Training & Model Files

### NNUE Model
- **File:** `src/nnue/halfkp_int8.pt` (8-bit quantized TorchScript model)
- **Architecture:** HalfKP input (40,960 features) → hidden layers → single evaluation output
- **Training Data:** 316 million Stockfish-evaluated positions
- **Training Script:** `src/nnue/nnue_train.py`

### Dataset Processing
For training with large datasets (300M+ positions):
```bash
cd src/nnue
python nnue_train.py --csv data.csv --epochs 5 --batch-size 32768
```

The pipeline:
1. Reads CSV in chunks (avoids loading entire dataset into RAM)
2. Computes HalfKP indices for each position
3. Caches indices to NPZ (skips CSV processing on subsequent runs)
4. Streams batches to GPU via DataLoader
5. Trains with mixed-precision and gradient clipping

### Book & Tablebase Files
- **Opening Book:** `src/book/book.json` (positions → evaluations, 20-ply depth)
- **Syzygy Tablebases:** Place in `src/endgame/syzygy/` (optional, 3-7 piece endgames)
- **Gaviota Tablebases:** Place in `src/endgame/gaviota/3/`, `4/`, `5/` (optional)

## Project Structure

```
Mini-Stockfish/
├── src/
│   ├── main.py              # Application entry point
│   ├── game.py              # Game state and rendering
│   ├── board.py             # Board representation
│   ├── ai.py                # AI engine interface
│   ├── core_search.pyx      # Optimized search (Cython)
│   ├── bitboard.py          # Bitboard operations
│   ├── accumulator.py       # NNUE accumulator
│   ├── piece.py             # Piece classes
│   ├── move.py              # Move representation
│   ├── square.py            # Square representation
│   ├── dragger.py           # Drag-and-drop handler
│   ├── promotion.py         # Pawn promotion modal
│   ├── theme.py             # UI themes
│   ├── config.py            # Configuration
│   ├── sound.py             # Sound effects
│   ├── color.py             # Color definitions
│   ├── const.py             # Constants
│   ├── nnue/
│   │   ├── halfkp.py        # HalfKP encoding
│   │   ├── nnue_train.py    # Training pipeline
│   │   ├── evaluate_model.py # Model evaluation
│   │   ├── check_model.py   # Model verification
│   │   ├── pth_to_pt.py     # Model conversion
│   │   └── halfkp_int8.pt   # Trained model
│   ├── book/
│   │   ├── preprocess.py    # Opening book preprocessing
│   │   └── book.json        # Opening book data
│   ├── endgame/
│   │   ├── pgn_preprocess.py # Endgame data processing
│   │   ├── syzygy/          # Syzygy tablebase files
│   │   └── gaviota/         # Gaviota tablebase files
│   ├── setup.py             # Cython compilation script
│   ├── requirements.txt      # Python dependencies
│   └── test_movegen.py      # Move generation tests
├── README.md
└── .claude/                 # Claude Code configuration

```

## Development Notes

### Building from Source
```bash
cd src
python setup.py build_ext --inplace
```

This compiles `core_search.pyx` to a C extension for performance.

### Running Tests
```bash
cd src
python test_movegen.py
```

Validates legal move generation and piece movement.

### Memory Files
The project tracks implementation notes in:
- `history_heuristic_root_cause.md` — Odd/even depth pruning analysis
- `killer_moves_implementation.md` — Move ordering optimization details

## Troubleshooting

### "Module not found: core_search"
Run `python setup.py build_ext --inplace` in the `src/` directory to compile the Cython extension.

### "CUDA out of memory"
Reduce batch size in `nnue_train.py` or switch to CPU-only mode by removing the CUDA installation.

### "Tablebase not found"
Optional endgame tables. The engine runs fine without them (falls back to NNUE evaluation).

### "No opening book"
The engine generates reasonable moves without a book, but includes one for better opening play.

## Performance Benchmarks

- **Search Speed:** ~1M-2M nodes per second (depth 10 on modern CPU)
- **NNUE Inference:** ~5,000-10,000 positions per second (CPU)
- **Move Generation:** ~500k-1M moves per second
- **Typical Rating:** ~1500-1800 ELO (depends on depth and hardware)

## Future Enhancements

- [ ] Multi-threaded search (parallel alpha-beta)
- [ ] Principal Variation Search (PVS) for better pruning
- [ ] Neural network training from game results
- [ ] Web-based UI (WebGL/Three.js)
- [ ] UCI protocol implementation for use with arena tools
- [ ] Automated architecture diagram generation

## References & Inspiration

- Initial structure based on [Chess Engine Tutorial](https://www.youtube.com/watch?v=OpL0Gcfn4B4)
- NNUE architecture from Stockfish's neural network evaluation
- Move ordering techniques from modern chess engines
- Alpha-beta pruning from classic game theory

## License

This project is for educational purposes. Feel free to modify and extend it for your own learning.

## Contributing

Contributions welcome! Focus areas:
1. Search optimization (node pruning, move ordering)
2. Neural network improvements (architecture, training data)
3. UI enhancements (themes, move annotations)
4. Performance profiling and optimization

---

**Last Updated:** 2026-05-31  
**Author:** Daris Chen  
**Version:** 1.0.0
