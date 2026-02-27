# accumulator.py

import numpy as np
import chess
from nnue.halfkp import halfkp_indices_for_fen, piece_to_idx, NUM_NONKING, PIECES_PER_KING

PAD_LEN = 30  # match TorchScript tracing pad length


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
        self._stack = []   # stack of (idx0_copy, idx1_copy) for rollback

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
        return self.idx0, self.idx1

    def update(self, move: chess.Move, captured: chess.Piece = None):
        """
        Update indices after the caller has already called board.push(move).
        `self.board` points to the same board, so it's already in post-move state.

        HalfKP index = king_sq * PIECES_PER_KING + piece_sq * NUM_NONKING + piece_type_offset
        Since piece_sq is encoded, every move changes the mover's index.
        """
        assert self.board is not None, "Call init() before update()"

        # Save snapshot before any changes
        self._stack.append((self.idx0.copy(), self.idx1.copy()))

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

        # Detect en passant: captured pawn but no piece was on to_sq before move.
        # The caller passes the actual captured piece. For EP, mover is a pawn
        # and captured is a pawn, but the captured pawn wasn't on to_sq.
        is_ep = (captured is not None
                 and captured.piece_type == chess.PAWN
                 and mover.piece_type == chess.PAWN
                 and abs(from_sq % 8 - to_sq % 8) == 1  # diagonal pawn move
                 and self.board.piece_at(to_sq) == mover)  # mover landed on to_sq, no piece was there before

        if is_ep:
            # Captured pawn was behind the to_square (from mover's perspective)
            if mover.color:  # white captured
                cap_sq = to_sq - 8
            else:
                cap_sq = to_sq + 8
        else:
            cap_sq = to_sq

        # For each view (view 0 = white king perspective, view 1 = black king perspective):
        for view_idx, view_arr in enumerate([self.idx0, self.idx1]):
            king_sq = self.board.king(bool(view_idx))

            # 1) Remove captured piece's index (if capture)
            if captured and captured.piece_type != chess.KING:
                cap_key = (captured.color, captured.piece_type)
                if cap_key in piece_to_idx:
                    old_cap_idx = king_sq * PIECES_PER_KING + cap_sq * NUM_NONKING + piece_to_idx[cap_key]
                    mask = (view_arr == old_cap_idx)
                    positions = np.where(mask)[0]
                    if len(positions) > 0:
                        view_arr[positions[0]] = 0

            # 2) Update mover's index: remove old (from_sq), add new (to_sq)
            mover_key = (mover.color, mover.piece_type)
            if mover_key in piece_to_idx:
                old_idx = king_sq * PIECES_PER_KING + from_sq * NUM_NONKING + piece_to_idx[mover_key]
                new_idx = king_sq * PIECES_PER_KING + to_sq * NUM_NONKING + piece_to_idx[mover_key]
                mask = (view_arr == old_idx)
                positions = np.where(mask)[0]
                if len(positions) > 0:
                    view_arr[positions[0]] = new_idx

        return self.idx0, self.idx1

    def rollback(self, move: chess.Move, captured: chess.Piece = None):
        """
        Undo the incremental update by restoring the saved snapshot.
        The caller has already called board.pop(), so self.board is back to pre-move state.
        O(1) — no FEN parsing or recomputation needed.
        """
        assert self.board is not None, "Call init() before rollback()"
        self.idx0, self.idx1 = self._stack.pop()
        return self.idx0, self.idx1

    def _recompute(self):
        """Full recompute of indices from current board state."""
        raw0, raw1 = halfkp_indices_for_fen(self.board.fen())
        self.idx0 = self._pad(raw0)
        self.idx1 = self._pad(raw1)
