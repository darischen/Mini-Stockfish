# accumulator.py

import numpy as np
import os
import torch
import chess
from numba import jit
from nnue.halfkp import halfkp_indices_for_fen, piece_to_idx, NUM_NONKING, PIECES_PER_KING

PAD_LEN = 30  # match TorchScript tracing pad length

@jit(nopython=True)
def evaluate_jit(hidden0, hidden1, fc2_w, fc2_b, fc3_w, fc3_b, out_w, out_b):
	"""JIT-compiled MLP evaluation on cached hidden activations."""
	h = np.concatenate((np.maximum(hidden0, 0), np.maximum(hidden1, 0)))
	x = fc2_w @ h + fc2_b
	x = np.maximum(x, 0)
	x = fc3_w @ x + fc3_b
	x = np.maximum(x, 0)
	out = out_w @ x + out_b
	return float(out[0])

@jit(nopython=True)
def update_view_jit(idx_arr, hidden, emb_w, old_cap_idx, old_idx, new_idx):
	"""Modify hidden and idx_arr in-place for one view. No return — arrays mutated directly."""
	if old_cap_idx >= 0:
		hidden -= emb_w[old_cap_idx]
		hidden += emb_w[0]
		for i in range(len(idx_arr)):
			if idx_arr[i] == old_cap_idx:
				idx_arr[i] = 0
				break
	hidden -= emb_w[old_idx]
	hidden += emb_w[new_idx]
	for i in range(len(idx_arr)):
		if idx_arr[i] == old_idx:
			idx_arr[i] = new_idx
			break

# --- Module-level weight cache (loaded once) ---
_weights = None

def _load_weights():
    """Load HalfKP model weights from checkpoint into numpy arrays."""
    global _weights
    if _weights is not None:
        return _weights

    base = os.path.dirname(__file__)
    pth_path = os.path.join(base, "nnue", "halfkp_best.pth")

    state = torch.load(pth_path, map_location="cpu", weights_only=True)
    _weights = {
        'emb0': state['emb0.weight'].numpy().astype(np.float32),    # (40960, 1024)
        'emb1': state['emb1.weight'].numpy().astype(np.float32),    # (40960, 1024)
        'fc2_w': state['fc2.weight'].numpy().astype(np.float32),    # (64, 2048)
        'fc2_b': state['fc2.bias'].numpy().astype(np.float32),      # (64,)
        'fc3_w': state['fc3.weight'].numpy().astype(np.float32),    # (64, 64)
        'fc3_b': state['fc3.bias'].numpy().astype(np.float32),      # (64,)
        'out_w': state['fc_out.weight'].numpy().astype(np.float32), # (1, 64)
        'out_b': state['fc_out.bias'].numpy().astype(np.float32),   # (1,)
    }
    return _weights


class Accumulator:
    """
    Maintains HalfKP index arrays (idx0, idx1) for a python-chess.Board,
    updating incrementally on push/pop.
    Uses a snapshot stack so rollback never needs to recompute from FEN.

    The caller (search) owns the board and calls board.push()/board.pop().
    This class only receives a reference to the board — it does NOT push/pop itself.
    """

    def __init__(self):
        self.idx0 = None   # numpy int64 array, padded to PAD_LEN
        self.idx1 = None
        self.board = None
        self._stack = []   # stack of (idx0, idx1, hidden0, hidden1) for rollback
        # Cached hidden activations (accumulated embedding sums)
        self.hidden0 = None  # shape (1024,) - view 0 (white king perspective)
        self.hidden1 = None  # shape (1024,) - view 1 (black king perspective)

    def _pad(self, lst):
        """Pad an index list to PAD_LEN with zeros and return int64 numpy array."""
        padded = lst + [0] * (PAD_LEN - len(lst))
        return np.array(padded[:PAD_LEN], dtype=np.int64)

    def init(self, board: chess.Board):
        """
        Initialize the accumulator from scratch.
        `board` is a reference — the caller manages push/pop on it.
        """
        self.board = board
        self._stack = []
        raw0, raw1 = halfkp_indices_for_fen(board.fen())
        self.idx0 = self._pad(raw0)
        self.idx1 = self._pad(raw1)

        # Compute initial hidden activations by summing embeddings
        # Use padded idx arrays (not raw) to match the model's forward pass,
        # where padding zeros contribute emb[0] to the sum.
        w = _load_weights()
        self.hidden0 = w['emb0'][self.idx0].sum(axis=0).copy()
        self.hidden1 = w['emb1'][self.idx1].sum(axis=0).copy()

        return self.idx0, self.idx1

    def update(self, move: chess.Move, captured: chess.Piece = None, old_ep_square=None):
        """
        Update indices after the caller has already called board.push(move).
        `self.board` points to the same board, so it's already in post-move state.

        HalfKP index = king_sq * PIECES_PER_KING + piece_sq * NUM_NONKING + piece_type_offset
        Since piece_sq is encoded, every move changes the mover's index.

        `old_ep_square` is the board's ep_square BEFORE the push (avoids pop/push round-trip).
        """
        assert self.board is not None, "Call init() before update()"

        # Save snapshot before any changes (including hidden activations)
        self._stack.append((self.idx0.copy(), self.idx1.copy(),
                            self.hidden0.copy(), self.hidden1.copy()))

        # Null move (0000) — just flip turn, no pieces change
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

        # Detect en passant using pre-push ep_square (avoids expensive pop/push round-trip)
        is_ep = (old_ep_square is not None
                 and to_sq == old_ep_square
                 and mover is not None
                 and mover.piece_type == chess.PAWN)

        if is_ep:
            # Captured pawn was behind the to_square (from mover's perspective)
            if mover.color:  # white captured
                cap_sq = to_sq - 8
            else:
                cap_sq = to_sq + 8
        else:
            cap_sq = to_sq

        w = _load_weights()
        mover_key = (mover.color, mover.piece_type)

        # For each view (view 0 = white king perspective, view 1 = black king perspective):
        for view_idx, view_arr, hidden, emb_w in (
            (0, self.idx0, self.hidden0, w['emb0']),
            (1, self.idx1, self.hidden1, w['emb1']),
        ):
            king_sq = self.board.king(bool(view_idx))

            # Resolve capture index (-1 = no capture)
            old_cap_idx = -1
            if captured and captured.piece_type != chess.KING:
                cap_key = (captured.color, captured.piece_type)
                if cap_key in piece_to_idx:
                    old_cap_idx = king_sq * PIECES_PER_KING + cap_sq * NUM_NONKING + piece_to_idx[cap_key]

            # Resolve mover indices and dispatch to JIT
            if mover_key in piece_to_idx:
                old_idx = king_sq * PIECES_PER_KING + from_sq * NUM_NONKING + piece_to_idx[mover_key]
                new_idx = king_sq * PIECES_PER_KING + to_sq * NUM_NONKING + piece_to_idx[mover_key]
                update_view_jit(view_arr, hidden, emb_w, old_cap_idx, old_idx, new_idx)
            elif old_cap_idx >= 0:
                # Capture only (mover not in piece_to_idx — shouldn't happen, but be safe)
                update_view_jit(view_arr, hidden, emb_w, old_cap_idx, 0, 0)

        return self.idx0, self.idx1

    def rollback(self, move: chess.Move, captured: chess.Piece = None):
        """
        Undo the incremental update by restoring the saved snapshot.
        The caller has already called board.pop(), so self.board is back to pre-move state.
        O(1) — no FEN parsing or recomputation needed.
        """
        assert self.board is not None, "Call init() before rollback()"
        self.idx0, self.idx1, self.hidden0, self.hidden1 = self._stack.pop()
        return self.idx0, self.idx1

    def evaluate(self):
        """
        Run the MLP on cached hidden activations.
        Returns the network's scalar output (from White's perspective).

        Computation: ~135k ops vs ~921k for full forward pass.
        """
        w = _load_weights()
        return evaluate_jit(self.hidden0, self.hidden1,
                           w['fc2_w'], w['fc2_b'],
                           w['fc3_w'], w['fc3_b'],
                           w['out_w'], w['out_b'])

    def _recompute(self):
        """Full recompute of indices and hidden activations from current board state."""
        raw0, raw1 = halfkp_indices_for_fen(self.board.fen())
        self.idx0 = self._pad(raw0)
        self.idx1 = self._pad(raw1)
        w = _load_weights()
        self.hidden0 = w['emb0'][self.idx0].sum(axis=0).copy()
        self.hidden1 = w['emb1'][self.idx1].sum(axis=0).copy()
