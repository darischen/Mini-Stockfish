# accumulator.py
import torch
import numpy as np
import chess
from nnue.nnue_train import fen_to_features

class Accumulator:
    """
    Incrementally maintains a 771‑dim feature vector for a python‑chess.Board,
    updating only the entries that change on each push/pop rather than recomputing
    the full vector every time.
    """

    # map python‑chess piece_type → channel 0–5
    _piece_base = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK:   3,
        chess.QUEEN:  4,
        chess.KING:   5,
    }

    # adjacency offsets for king‑safety features
    _king_offsets = [-9, -8, -7, -1, 1, 7, 8, 9]
    
    _phase_weights = {
        chess.PAWN:   0,
        chess.KNIGHT: 1,
        chess.BISHOP: 1,
        chess.ROOK:   2,
        chess.QUEEN:  4,
        chess.KING:   0,
    }
    
    _max_phase = sum(w * 2 for w in _phase_weights.values())

    def __init__(self):
        self.state = None  # torch.Tensor of shape [1,771]
        self.board = None  # the python‑chess.Board we’re tracking

    def init(self, board: chess.Board):
        """
        Initialize the accumulator from scratch.
        :param board: a python‑chess.Board at the root of your search
        :return: a [1×771] torch.Tensor ready to feed into your NNUEModel
        """
        # full‑vector fallback via fen_to_features
        arr = fen_to_features(board.fen())           # numpy array shape (771,)
        self.state = torch.from_numpy(arr).unsqueeze(0)  # [1,771]
        self.board = board.copy()                    # keep our own copy
        return self.state

    def update(self, move: chess.Move, captured: chess.Piece = None):
        """
        Apply a move incrementally to the internal feature vector.
        Assumes you've already done `self.board.push(move)`.
        :param move: the chess.Move just applied to self.board
        :param captured: the Piece that was on move.to_square before the push (or None)
        :return: updated [1×771] tensor
        """
        assert self.state is not None and self.board is not None, \
            "Call init() before update()"

        # attacker is now on to_square
        self.board.push(move)
        atk = self.board.piece_at(move.to_square)
        color_offset = 0 if atk.color == chess.WHITE else 6
        base_idx = self._piece_base[atk.piece_type]
        ch = base_idx + color_offset

        # clear old one‑hot at from_square
        frm = move.from_square * 12 + ch
        to  = move.to_square  * 12 + ch
        self.state[0, frm] = 0.0
        self.state[0, to]  = 1.0

        # clear victim’s one‑hot if this was a capture
        if captured:
            cap_ch = self._piece_base[captured.piece_type] + \
                     (0 if captured.color == chess.WHITE else 6)
            cap_idx = move.to_square * 12 + cap_ch
            self.state[0, cap_idx] = 0.0

        # recompute just the extra 3 features: king‑safety & mobility
        extras = self._compute_extras()
        # indices 768,769,770
        self.state[0, 768:] = extras

        return self.state

    def rollback(self, move: chess.Move, captured: chess.Piece = None):
        """
        Undo the incremental update. Call *after* you do `self.board.pop()`.
        Mirrors `update` but swaps from/to and re‑inserts the captured piece.
        """
        # swap attacker back
        self.board.pop()
        atk = self.board.piece_at(move.from_square)
        color_offset = 0 if atk.color == chess.WHITE else 6
        base_idx = self._piece_base[atk.piece_type]
        ch = base_idx + color_offset

        frm = move.from_square * 12 + ch
        to  = move.to_square  * 12 + ch
        self.state[0, frm] = 1.0
        self.state[0, to]  = 0.0

        # restore victim one‑hot if needed
        if captured:
            cap_ch = self._piece_base[captured.piece_type] + \
                     (0 if captured.color == chess.WHITE else 6)
            cap_idx = move.to_square * 12 + cap_ch
            self.state[0, cap_idx] = 1.0

        # recompute extras
        extras = self._compute_extras()
        self.state[0, 768:] = extras

        return self.state

    def _compute_extras(self):
        """
        Recompute the 3 extra features on the current self.board:
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
                    # avoid file‑wrap
                    if abs((nb % 8) - (sq % 8)) > 1:
                        continue
                    p = bb.piece_at(nb)
                    if p and p.piece_type == chess.PAWN and p.color == color:
                        ks += 1.0
            return ks

        white_ks = king_safety(chess.WHITE)
        black_ks = king_safety(chess.BLACK)
        # mobility = number of legal moves
        mob = float(len(list(bb.legal_moves)))

        return torch.tensor([white_ks, black_ks, mob], dtype=torch.float32)

    def game_phase(self) -> float:
        """
        Compute the endgame fraction (0 = opening/middlegame, 1 = endgame) based on piece counts.
        """
        phase = 0
        for pc in self.board.piece_map().values():
            phase += self._phase_weights.get(pc.piece_type, 0)
        # clamp into [0,1]
        return min(max(phase / self._max_phase, 0.0), 1.0)