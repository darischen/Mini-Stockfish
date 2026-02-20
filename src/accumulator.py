# accumulator.py

import numpy as np
import chess
from nnue.halfkp import halfkp_indices_for_fen, piece_to_idx, NUM_NONKING

PAD_LEN = 30  # match TorchScript tracing pad length


class Accumulator:
    """
    Maintains HalfKP index arrays (idx0, idx1) for a python-chess.Board,
    updating incrementally on push/pop.
    """

    def __init__(self):
        self.idx0 = None   # numpy int64 array, padded to PAD_LEN
        self.idx1 = None
        self.board = None

    def _pad(self, lst):
        """Pad an index list to PAD_LEN with zeros and return int64 numpy array."""
        padded = lst + [0] * (PAD_LEN - len(lst))
        return np.array(padded[:PAD_LEN], dtype=np.int64)

    def init(self, board: chess.Board):
        """
        Initialize the accumulator from scratch.
        """
        self.board = board.copy()
        raw0, raw1 = halfkp_indices_for_fen(board.fen())
        self.idx0 = self._pad(raw0)
        self.idx1 = self._pad(raw1)
        return self.idx0, self.idx1

    def update(self, move: chess.Move, captured: chess.Piece = None):
        """
        Apply a move incrementally. Call before or after board.push(move).
        For king moves, we do a full recompute since all indices depend on king square.
        """
        assert self.board is not None, "Call init() before update()"

        self.board.push(move)

        mover = self.board.piece_at(move.to_square)

        # King move → full recompute (all HalfKP indices depend on king square)
        if mover and mover.piece_type == chess.KING:
            self._recompute()
            return self.idx0, self.idx1

        # Promotion → full recompute (piece type changed)
        if move.promotion:
            self._recompute()
            return self.idx0, self.idx1

        # Capture → remove captured piece's index from both views
        if captured and captured.piece_type != chess.KING:
            key = (captured.color, captured.piece_type)
            if key in piece_to_idx:
                # Remove one instance of the captured piece index from each view
                for view_idx, view_arr in enumerate([self.idx0, self.idx1]):
                    king_sq = self.board.king(bool(view_idx))
                    cap_idx = king_sq * NUM_NONKING + piece_to_idx[key]
                    # Find first occurrence and zero it out
                    mask = (view_arr == cap_idx)
                    positions = np.where(mask)[0]
                    if len(positions) > 0:
                        view_arr[positions[0]] = 0

        # Normal non-king move: HalfKP index = king_sq * 10 + piece_offset
        # Since the piece's own square is NOT encoded, indices don't change
        return self.idx0, self.idx1

    def rollback(self, move: chess.Move, captured: chess.Piece = None):
        """
        Undo the incremental update. Call after board.pop().
        Safe fallback: full recompute from FEN.
        """
        assert self.board is not None, "Call init() before rollback()"
        self.board.pop()
        self._recompute()
        return self.idx0, self.idx1

    def _recompute(self):
        """Full recompute of indices from current board state."""
        raw0, raw1 = halfkp_indices_for_fen(self.board.fen())
        self.idx0 = self._pad(raw0)
        self.idx1 = self._pad(raw1)
