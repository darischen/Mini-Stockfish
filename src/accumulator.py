# accumulator_py.py

import numpy as np
import chess
from nnue.nnue_train import fen_to_features

class Accumulator:
    """
    Incrementally maintains a 771-dim NumPy feature vector for a python-chess.Board,
    updating only the entries that change on each push/pop.
    """

    # map python-chess piece_type → channel 0–5
    _piece_base = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }

    # adjacency offsets for king-safety features
    _king_offsets = [-9, -8, -7, -1, 1, 7, 8, 9]

    def __init__(self):
        self.state = None    # numpy.ndarray shape (771,), dtype float32
        self.board = None    # python-chess.Board we’re tracking

    def init(self, board: chess.Board) -> np.ndarray:
        """
        Initialize the accumulator from scratch.
        :param board: a python-chess.Board at the root of your search
        :return: a 1×771 NumPy array
        """
        # full-vector fallback via fen_to_features (returns np.float32[771])
        arr = fen_to_features(board.fen())  
        self.state = arr.copy()            # keep our own copy
        self.board = board.copy()
        return self.state

    def update(self, move: chess.Move, captured: chess.Piece = None) -> np.ndarray:
        """
        Apply a move incrementally to the internal feature vector.
        Assumes you've already done `board.push(move)`.
        """
        assert self.state is not None and self.board is not None, \
            "Call init() before update()"

        # 1) push to our copy
        self.board.push(move)

        # 2) attacker channel index
        atk = self.board.piece_at(move.to_square)
        color_offset = 0 if atk.color == chess.WHITE else 6
        ch = self._piece_base[atk.piece_type] + color_offset

        # 3) clear old one-hot at from_square; set at to_square
        frm_idx = move.from_square * 12 + ch
        to_idx  = move.to_square   * 12 + ch
        self.state[frm_idx] = 0.0
        self.state[to_idx]  = 1.0

        # 4) clear victim’s one-hot if this was a capture
        if captured:
            cap_ch  = self._piece_base[captured.piece_type] + \
                      (0 if captured.color == chess.WHITE else 6)
            cap_idx = move.to_square * 12 + cap_ch
            self.state[cap_idx] = 0.0

        # 5) recompute only the 3 extra features
        self.state[768:771] = self._compute_extras()
        return self.state

    def rollback(self, move: chess.Move, captured: chess.Piece = None) -> np.ndarray:
        """
        Undo the incremental update. Call *after* you do `board.pop()`.
        """
        assert self.state is not None and self.board is not None, \
            "Call init() before rollback()"

        # 1) pop on our copy
        self.board.pop()

        # 2) attacker goes back from to -> from
        atk = self.board.piece_at(move.from_square)
        color_offset = 0 if atk.color == chess.WHITE else 6
        ch = self._piece_base[atk.piece_type] + color_offset

        frm_idx = move.from_square * 12 + ch
        to_idx  = move.to_square   * 12 + ch
        self.state[frm_idx] = 1.0
        self.state[to_idx]  = 0.0

        # 3) restore victim if needed
        if captured:
            cap_ch  = self._piece_base[captured.piece_type] + \
                      (0 if captured.color == chess.WHITE else 6)
            cap_idx = move.to_square * 12 + cap_ch
            self.state[cap_idx] = 1.0

        # 4) recompute extras again
        self.state[768:771] = self._compute_extras()
        return self.state

    def _compute_extras(self) -> np.ndarray:
        """
        Compute the 3 extra features on the current self.board:
          [white_king_safety, black_king_safety, mobility]
        """
        bb = self.board

        def king_safety(color):
            ks = 0.0
            sq = bb.king(color)
            if sq is None:
                return ks
            for off in self._king_offsets:
                nb = sq + off
                if 0 <= nb < 64:
                    # avoid file-wrap
                    if abs((nb % 8) - (sq % 8)) > 1:
                        continue
                    p = bb.piece_at(nb)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        ks += 1.0
            return ks

        white_ks = king_safety(chess.WHITE)
        black_ks = king_safety(chess.BLACK)
        mob      = float(len(list(bb.legal_moves)))

        return np.array([white_ks, black_ks, mob], dtype=np.float32)
