# Incremental NNUE Activation Caching Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cache NNUE hidden layer activations in the Accumulator and update them incrementally on each move, so only the tiny MLP (2048->64->64->1) runs per evaluation instead of the full embedding sum (~786k ops).

**Architecture:** Load embedding weights and MLP weights from `halfkp_best.pth` into numpy arrays at startup. The Accumulator maintains two 1024-dim vectors (`hidden0`, `hidden1`) representing the summed embeddings for each king perspective. On non-king moves, subtract the old piece embedding and add the new one (~4k ops). On evaluation, clamp the hidden vectors and run the MLP in numpy (~135k ops). King moves and promotions trigger a full recompute. The snapshot stack (already used for index rollback) is extended to include the hidden vectors.

**Tech Stack:** Python, NumPy, PyTorch (weight loading only), Cython

---

## Chunk 1: Accumulator with Incremental Activations

### Task 1: Add weight loading and hidden state to Accumulator

**Files:**
- Modify: `src/accumulator.py`

The `Accumulator` class currently stores only HalfKP index arrays (`idx0`, `idx1`). We need to:
1. Load model weights (embeddings + MLP layers) from `halfkp_best.pth` once at module level
2. Add `hidden0` / `hidden1` fields (1024-dim numpy arrays) for cached activation sums
3. Include hidden vectors in the snapshot stack for rollback
4. Add an `evaluate()` method that runs the MLP on cached activations

Key constants from `src/nnue/halfkp.py`:
- `TABLE_SIZE = 40960` (embedding table size)
- `HIDDEN_SIZE = 1024` (embedding dimension)
- `MLP_HIDDEN = 64` (MLP hidden size)
- Model layers: `emb0(40960, 1024)`, `emb1(40960, 1024)`, `fc2(2048, 64)`, `fc3(64, 64)`, `fc_out(64, 1)`

- [ ] **Step 1: Add module-level weight loading**

At the top of `accumulator.py`, after imports, add a function that loads weights from the `.pth` checkpoint into numpy arrays. This runs once when the module is imported.

```python
import torch
import os

# --- Module-level weight cache (loaded once) ---
_weights = None

def _load_weights():
    """Load HalfKP model weights from checkpoint into numpy arrays."""
    global _weights
    if _weights is not None:
        return _weights

    base = os.path.dirname(__file__)
    pth_path = os.path.join(base, "nnue", "halfkp_best.pth")

    state = torch.load(pth_path, map_location="cpu")
    _weights = {
        'emb0': state['emb0.weight'].numpy(),      # (40960, 1024)
        'emb1': state['emb1.weight'].numpy(),      # (40960, 1024)
        'fc2_w': state['fc2.weight'].numpy(),       # (64, 2048)
        'fc2_b': state['fc2.bias'].numpy(),         # (64,)
        'fc3_w': state['fc3.weight'].numpy(),       # (64, 64)
        'fc3_b': state['fc3.bias'].numpy(),         # (64,)
        'out_w': state['fc_out.weight'].numpy(),    # (1, 64)
        'out_b': state['fc_out.bias'].numpy(),      # (1,)
    }
    return _weights
```

- [ ] **Step 2: Modify `__init__` to add hidden state fields**

```python
def __init__(self):
    self.idx0 = None
    self.idx1 = None
    self.board = None
    self._stack = []
    # Cached hidden activations (accumulated embedding sums)
    self.hidden0 = None  # shape (1024,) - view 0 (white king perspective)
    self.hidden1 = None  # shape (1024,) - view 1 (black king perspective)
```

- [ ] **Step 3: Modify `init()` to compute initial hidden activations**

When initializing from scratch, sum the embeddings for all active pieces:

```python
def init(self, board: chess.Board):
    self.board = board
    self._stack = []
    raw0, raw1 = halfkp_indices_for_fen(board.fen())
    self.idx0 = self._pad(raw0)
    self.idx1 = self._pad(raw1)

    # Compute initial hidden activations by summing embeddings
    w = _load_weights()
    self.hidden0 = w['emb0'][raw0].sum(axis=0).copy()  # sum embeddings for view 0
    self.hidden1 = w['emb1'][raw1].sum(axis=0).copy()  # sum embeddings for view 1

    return self.idx0, self.idx1
```

- [ ] **Step 4: Modify `update()` to incrementally update hidden vectors**

The key optimization. On non-king, non-promotion moves, update the hidden vectors by subtracting the removed embedding and adding the new one:

```python
def update(self, move: chess.Move, captured: chess.Piece = None):
    assert self.board is not None, "Call init() before update()"

    # Save snapshot including hidden vectors
    self._stack.append((self.idx0.copy(), self.idx1.copy(),
                        self.hidden0.copy(), self.hidden1.copy()))

    # Null move — no pieces change
    if not move:
        return self.idx0, self.idx1

    mover = self.board.piece_at(move.to_square)

    # King move → full recompute (all HalfKP indices depend on king square)
    if mover and mover.piece_type == chess.KING:
        self._recompute()
        return self.idx0, self.idx1

    # Promotion → full recompute (piece type changed)
    if move.promotion:
        self._recompute()
        return self.idx0, self.idx1

    from_sq = move.from_square
    to_sq = move.to_square

    # Detect en passant
    is_ep = (captured is not None
             and captured.piece_type == chess.PAWN
             and mover.piece_type == chess.PAWN
             and abs(from_sq % 8 - to_sq % 8) == 1
             and self.board.piece_at(to_sq) == mover)

    if is_ep:
        cap_sq = to_sq - 8 if mover.color else to_sq + 8
    else:
        cap_sq = to_sq

    w = _load_weights()
    emb_weights = [w['emb0'], w['emb1']]
    hiddens = [self.hidden0, self.hidden1]

    for view_idx in range(2):
        view_arr = self.idx0 if view_idx == 0 else self.idx1
        king_sq = self.board.king(bool(view_idx))
        emb_w = emb_weights[view_idx]
        hidden = hiddens[view_idx]

        # 1) Remove captured piece's embedding
        if captured and captured.piece_type != chess.KING:
            cap_key = (captured.color, captured.piece_type)
            if cap_key in piece_to_idx:
                old_cap_idx = king_sq * PIECES_PER_KING + cap_sq * NUM_NONKING + piece_to_idx[cap_key]
                # Update hidden: subtract captured piece's embedding
                hidden -= emb_w[old_cap_idx]
                # Update index array
                mask = (view_arr == old_cap_idx)
                positions = np.where(mask)[0]
                if len(positions) > 0:
                    view_arr[positions[0]] = 0

        # 2) Update mover's embedding: remove old, add new
        mover_key = (mover.color, mover.piece_type)
        if mover_key in piece_to_idx:
            old_idx = king_sq * PIECES_PER_KING + from_sq * NUM_NONKING + piece_to_idx[mover_key]
            new_idx = king_sq * PIECES_PER_KING + to_sq * NUM_NONKING + piece_to_idx[mover_key]
            # Update hidden: subtract old position, add new position
            hidden -= emb_w[old_idx]
            hidden += emb_w[new_idx]
            # Update index array
            mask = (view_arr == old_idx)
            positions = np.where(mask)[0]
            if len(positions) > 0:
                view_arr[positions[0]] = new_idx

    return self.idx0, self.idx1
```

- [ ] **Step 5: Modify `rollback()` to restore hidden vectors**

```python
def rollback(self, move: chess.Move, captured: chess.Piece = None):
    assert self.board is not None, "Call init() before rollback()"
    self.idx0, self.idx1, self.hidden0, self.hidden1 = self._stack.pop()
    return self.idx0, self.idx1
```

- [ ] **Step 6: Modify `_recompute()` to also recompute hidden vectors**

```python
def _recompute(self):
    """Full recompute of indices and hidden activations from current board state."""
    raw0, raw1 = halfkp_indices_for_fen(self.board.fen())
    self.idx0 = self._pad(raw0)
    self.idx1 = self._pad(raw1)
    w = _load_weights()
    self.hidden0 = w['emb0'][raw0].sum(axis=0).copy() if raw0 else np.zeros(1024, dtype=np.float32)
    self.hidden1 = w['emb1'][raw1].sum(axis=0).copy() if raw1 else np.zeros(1024, dtype=np.float32)
```

- [ ] **Step 7: Add `evaluate()` method**

This runs only the MLP layers on cached hidden activations. This is the fast path that replaces the full TorchScript forward pass:

```python
def evaluate(self):
    """
    Run the MLP on cached hidden activations.
    Returns the network's scalar output (from White's perspective).

    Computation: ~135k ops vs ~921k for full forward pass.
    """
    w = _load_weights()
    # Clamp (ReLU on accumulated sums, matching model's forward())
    h = np.concatenate([np.maximum(self.hidden0, 0),
                        np.maximum(self.hidden1, 0)])
    # MLP: fc2 -> relu -> fc3 -> relu -> fc_out
    x = w['fc2_w'] @ h + w['fc2_b']
    np.maximum(x, 0, out=x)  # relu in-place
    x = w['fc3_w'] @ x + w['fc3_b']
    np.maximum(x, 0, out=x)  # relu in-place
    out = w['out_w'] @ x + w['out_b']
    return float(out[0])
```

- [ ] **Step 8: Verify weight shapes match by adding a debug print to `_load_weights`**

Add a temporary print to `_load_weights()` after loading, to confirm shapes:

```python
print(f"[NNUE weights] emb0={_weights['emb0'].shape}, emb1={_weights['emb1'].shape}, "
      f"fc2={_weights['fc2_w'].shape}, fc3={_weights['fc3_w'].shape}, "
      f"out={_weights['out_w'].shape}")
```

Run: `cd src && python -c "from accumulator import Accumulator; a = Accumulator()"`
Expected: `[NNUE weights] emb0=(40960, 1024), emb1=(40960, 1024), fc2=(64, 2048), fc3=(64, 64), out=(1, 64)`

Remove the debug print after confirming.

- [ ] **Step 9: Commit**

```bash
git add src/accumulator.py
git commit -m "feat: add incremental activation caching to Accumulator

Load model weights from halfkp_best.pth into numpy arrays.
Maintain hidden0/hidden1 vectors (accumulated embedding sums).
Update incrementally on non-king moves, full recompute on king moves.
Add evaluate() method that runs just the MLP (~135k ops vs ~921k)."
```

---

### Task 2: Wire up cached evaluation in Cython search

**Files:**
- Modify: `src/core_search.pyx`

Replace calls to `nnue_eval_halfkp_py(acc.idx0, acc.idx1)` (which does a full TorchScript forward pass) with `acc.evaluate()` (which uses cached hidden activations + tiny MLP).

There are two locations where NNUE eval is called:
1. `quiesce()` at line 642: stand-pat evaluation
2. Nowhere else — `minimax()` only calls NNUE through `quiesce()`

- [ ] **Step 1: Modify quiesce() to use acc.evaluate()**

In `core_search.pyx`, change the NNUE eval block in `quiesce()` (around line 641-645):

Old code:
```python
if USE_NNUE:
    val = nnue_eval_halfkp_py(acc.idx0, acc.idx1)
    # NNUE returns from White's perspective; negamax needs side-to-move's
    if not board.turn:  # Black to move -> flip
        val = -val
```

New code:
```python
if USE_NNUE:
    val = acc.evaluate()
    # evaluate() returns from White's perspective; negamax needs side-to-move's
    if not board.turn:  # Black to move -> flip
        val = -val
```

- [ ] **Step 2: Rebuild the Cython extension**

Run:
```bash
cd src && python setup.py build_ext --inplace
```

Expected: Successful build producing `core_search.cp310-win_amd64.pyd`

- [ ] **Step 3: Smoke test — run a short game**

Run the engine briefly to verify it works:
```bash
cd src && python -c "
import chess
from accumulator import Accumulator
import core_search

core_search.init_nnue('nnue/halfkp_int8.pt')
core_search.set_use_nnue(True)

board = chess.Board()
acc = Accumulator()
acc.init(board)

# Test evaluate() directly
val = acc.evaluate()
print(f'Start position eval: {val:.4f}')

# Test a move + incremental update + eval
move = chess.Move.from_uci('e2e4')
captured = board.piece_at(move.to_square)
board.push(move)
acc.update(move, captured)
val2 = acc.evaluate()
print(f'After e4 eval: {val2:.4f}')

# Test rollback
board.pop()
acc.rollback(move, captured)
val3 = acc.evaluate()
print(f'After rollback eval: {val3:.4f}')
assert abs(val - val3) < 1e-6, f'Rollback mismatch: {val} vs {val3}'
print('Rollback OK')
"
```

Expected: Three eval values printed, rollback assertion passes.

- [ ] **Step 4: Correctness test — compare cached vs full eval on multiple positions**

Verify the cached incremental eval matches a full recompute for several moves:

```bash
cd src && python -c "
import chess
import numpy as np
from accumulator import Accumulator

board = chess.Board()
acc = Accumulator()
acc.init(board)

# Play 10 random-ish moves and compare incremental vs full recompute
moves = ['e2e4','e7e5','g1f3','b8c6','f1c4','g8f6','d2d3','d7d6','c1g5','f8e7']
for uci_str in moves:
    move = chess.Move.from_uci(uci_str)
    captured = board.piece_at(move.to_square)
    board.push(move)
    acc.update(move, captured)

    # Get incremental eval
    inc_val = acc.evaluate()

    # Full recompute for comparison
    acc2 = Accumulator()
    acc2.init(board)
    full_val = acc2.evaluate()

    diff = abs(inc_val - full_val)
    status = 'OK' if diff < 1e-4 else 'MISMATCH'
    print(f'{uci_str}: inc={inc_val:.4f} full={full_val:.4f} diff={diff:.6f} {status}')
    assert diff < 1e-4, f'Eval mismatch after {uci_str}: {inc_val} vs {full_val}'

print('All moves match!')
"
```

Expected: All 10 moves print OK, no assertion failures.

- [ ] **Step 5: Commit**

```bash
git add src/core_search.pyx
git commit -m "feat: use cached activations for NNUE eval in search

Replace nnue_eval_halfkp_py() call in quiesce() with acc.evaluate().
This skips the full embedding lookup+sum (~786k ops) and runs only
the MLP (~135k ops) for non-king-move positions."
```

---

### Task 3: Performance validation

**Files:**
- No file changes — benchmarking only

- [ ] **Step 1: Benchmark old vs new eval speed**

Time the evaluation per call:

```bash
cd src && python -c "
import time, chess
from accumulator import Accumulator
import core_search

core_search.init_nnue('nnue/halfkp_int8.pt')

board = chess.Board()
acc = Accumulator()
acc.init(board)

# Benchmark acc.evaluate() (new path)
N = 10000
start = time.perf_counter()
for _ in range(N):
    acc.evaluate()
elapsed_new = time.perf_counter() - start
print(f'acc.evaluate(): {elapsed_new/N*1e6:.1f} us/call ({N} calls in {elapsed_new:.3f}s)')

# Benchmark nnue_eval_halfkp_py (old path)
start = time.perf_counter()
for _ in range(N):
    core_search.nnue_eval_halfkp_py(acc.idx0, acc.idx1)
elapsed_old = time.perf_counter() - start
print(f'nnue_eval_halfkp_py(): {elapsed_old/N*1e6:.1f} us/call ({N} calls in {elapsed_old:.3f}s)')

print(f'Speedup: {elapsed_old/elapsed_new:.1f}x')
"
```

Expected: `acc.evaluate()` should be significantly faster than `nnue_eval_halfkp_py()`.

- [ ] **Step 2: Run a depth-4 search and compare timings**

```bash
cd src && python -c "
import time, chess
from accumulator import Accumulator
import core_search

core_search.init_nnue('nnue/halfkp_int8.pt')
core_search.set_use_nnue(True)

board = chess.Board()
acc = Accumulator()
acc.init(board)

key = core_search.compute_hash(board)
core_search.clear_tt()
core_search.reset_counters()

start = time.perf_counter()
val = core_search.minimax(board, acc, 4, -1e9, 1e9, 'white', key, 4)
elapsed = time.perf_counter() - start
nodes = core_search.get_nodes_evaluated()

print(f'Depth 4: val={val:.4f}, nodes={nodes}, time={elapsed:.2f}s, nps={nodes/elapsed:.0f}')
"
```

Expected: Noticeably faster than before (you can compare by temporarily reverting the `quiesce()` change).
