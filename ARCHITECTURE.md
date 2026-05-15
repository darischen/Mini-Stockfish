# Mini-Stockfish Architecture Map

Interactive visualization of the codebase structure, dependencies, and layers.

## Files

- **architecture.html** — Interactive visualization (open in browser)
- **architecture.json** — Dual-format data (node-link + hierarchical)
- **scripts/generate_architecture.py** — Tool to regenerate JSON from source code

## Opening the Map

```bash
# Open in default browser
open architecture.html
# or
firefox architecture.html
# or
google-chrome architecture.html
```

## Features

- **Search** — Filter modules by name in the sidebar or toolbar
- **Layer Filters** — Toggle visibility of UI, Game Logic, Search, Neural, Utilities layers
- **Click Nodes** — View module details and highlight dependencies
- **Zoom & Pan** — Mouse wheel to zoom, click-drag to pan the visualization
- **Layout** — Click "Re-Layout" to reorganize the graph
- **Reset** — Click "Reset View" to clear all filters and zoom

## Layer Structure

The architecture is organized into 5 layers:

- **UI Layer** (Light Blue) — User interface components
  - main.py, game.py, dragger.py, config.py, theme.py, sound.py, promotion.py, color.py

- **Game Logic Layer** (Light Green) — Chess rules and board representation
  - board.py, move.py, piece.py, square.py, const.py, bitboard.py

- **Search Engine Layer** (Orange) — Minimax algorithm with alpha-beta pruning
  - ai.py, core_search.pyx, accumulator.py

- **Neural Network Layer** (Light Purple) — NNUE evaluation and training
  - nnue_train.py, halfkp.py, check_model.py, evaluate_model.py

- **Utilities** (Light Gray) — Helper modules and data processing
  - book/preprocess.py, endgame/pgn_preprocess.py

## Module Statistics

- **Total Modules:** 23
- **Total Dependencies:** 33
- **Most Connected:** board.py (6 dependencies)
- **Entry Point:** main.py

## Regenerating the Map

If you add new modules or change imports, regenerate the JSON:

```bash
python scripts/generate_architecture.py
```

This script reads the module metadata and generates both node-link and hierarchical formats in `architecture.json`.

## How to Use for Development

1. **Understanding dependencies:** Click on a module to see what it imports and what imports it
2. **Finding related code:** Use search to find modules quickly
3. **Exploring by layer:** Toggle layers to focus on a specific architectural layer
4. **Discovering patterns:** Zoom out to see overall system structure, zoom in to see details

## Format Details

### Node-Link Format
Used by the D3.js visualization. Optimized for graph rendering with layers, components, nodes, and edges.

### Hierarchical Format
Organized as layers → components → modules. Better for navigation and documentation generation.

Both formats are in the same `architecture.json` file with root-level keys `nodeLink` and `hierarchical`.
