# ai.py
import math
import os
import time
import random
import torch  # Ensure PyTorch is installed if you plan to use a DNN
import torch.nn as nn
import chess  # Use python-chess for fast bitboard-backed move generation
import numpy as np
import bitboard  # Our bitboard module (square_bb, popcount, attacks, etc.)
from nnue.nnue_train import NNUEModel
from move import Move  # Move class for interoperability with the game engine
from square import Square

class ChessAI:
    def __init__(self, depth=3, use_dnn=False, model_path=None):
        """
        Initialize the chess AI.
        :param depth: How many plies to search.
        :param use_dnn: Whether to use a deep neural network (NNUE) for evaluation.
        :param model_path: Path to the pretrained model.
        """
        self.depth = depth
        self.use_dnn = use_dnn
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if self.use_dnn and model_path is not None and os.path.exists(model_path):
            self.model = NNUEModel(input_size=771).to(self.device)
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            self.model = None

    def choose_move(self, board, color: str):
        """
        Choose the best move for the given board and color.
        Uses python-chess internally (bitboard-backed) for fast move generation and push/pop,
        instead of mutating the game engine's Board instance.
        Returns a (piece, Move) tuple keyed back to the game engine.
        """
        # Initialize statistics
        self.nodes_evaluated = 0
        self.branches_pruned = 0

        # Convert current game state to a python-chess Board
        root_fen = board.get_fen()
        search_board = chess.Board(root_fen)
        search_board.turn = chess.BLACK

        best_move = None
        # For white we maximize, for black we minimize
        maximizing = (color == 'white')
        best_eval = -math.inf if maximizing else math.inf

        start = time.time()
        # Generate all legal moves via python-chess
        for uci in search_board.legal_moves:
            search_board.push(uci)
            val = self._minimax(search_board, self.depth - 1,
                                 -math.inf, math.inf,
                                 not maximizing, color)
            search_board.pop()

            if maximizing and val > best_eval:
                best_eval, best_move = val, uci
            elif not maximizing and val < best_eval:
                best_eval, best_move = val, uci

        elapsed = time.time() - start
        print(f"AI search complete. Nodes: {self.nodes_evaluated}, Pruned: {self.branches_pruned}, Time: {elapsed:.2f}s")
        print(f"Best eval for {color}: {best_eval:.4f}")

        if not best_move:
            return None
        # Map back to game-engine Move and piece
        src_idx = best_move.from_square
        dst_idx = best_move.to_square

        # unpack
        src_rank, src_file = divmod(src_idx, 8)
        dst_rank, dst_file = divmod(dst_idx, 8)

        # convert so that rank=0→row=7, rank=7→row=0
        initial = Square(7 - src_rank, src_file)
        final   = Square(7 - dst_rank, dst_file)
        mv = Move(initial, final)

        piece = board.squares[initial.row][initial.col].piece
        return (piece, mv)

    def _minimax(self, board: chess.Board, depth, alpha, beta, maximizing_player, ai_color):
        self.nodes_evaluated += 1
        if depth == 0 or board.is_game_over():
            return self._evaluate_bb(board, ai_color)

        if maximizing_player:
            max_eval = -math.inf
            # Order moves using simple MVV-LVA on python-chess board
            ordered = self._order_moves(bb=board, maximize=True)
            for uci in ordered:
                board.push(uci)
                val = self._minimax(board, depth - 1, alpha, beta, False, ai_color)
                board.pop()
                max_eval = max(max_eval, val)
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return max_eval
        else:
            min_eval = math.inf
            ordered = self._order_moves(bb=board, maximize=False)
            for uci in ordered:
                board.push(uci)
                val = self._minimax(board, depth - 1, alpha, beta, True, ai_color)
                board.pop()
                min_eval = min(min_eval, val)
                beta = min(beta, val)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return min_eval

    def _order_moves(self, bb: chess.Board, maximize: bool):
        # MVV-LVA: Victim value * 1000 - attacker value
        piece_vals = {'P':100,'N':300,'B':310,'R':400,'Q':900,'K':20000}
        scored = []
        for move in bb.legal_moves:
            victim = bb.piece_at(move.to_square)
            v_val = piece_vals.get(victim.symbol().upper(),0) if victim else 0
            attacker = bb.piece_at(move.from_square)
            a_val = piece_vals.get(attacker.symbol().upper(),0)
            score = 1000*v_val - a_val
            scored.append((move, score))
        scored.sort(key=lambda x: x[1], reverse=maximize)
        return [m for (m,_) in scored]

    def _evaluate_bb(self, bb: chess.Board, ai_color: str):
        """
        Bitboard-based static evaluation using PST interpolation.
        """
        # --- 1) Build per-piece bitboards from python-chess ---
        bitboards = {}
        for sq, pc in bb.piece_map().items():
            color = 'white' if pc.color else 'black'
            key = (color, pc.piece_type)
            bitboards[key] = bitboards.get(key, 0) | bitboard.square_bb(sq)

        # --- 2) Define material values and phase weights ---
        mat_values = {
            chess.PAWN:   100,
            chess.KNIGHT: 300,
            chess.BISHOP: 310,
            chess.ROOK:   400,
            chess.QUEEN:  900,
            chess.KING:   20000
        }
        phase_weights = {
            chess.PAWN:   0,
            chess.KNIGHT: 1,
            chess.BISHOP: 1,
            chess.ROOK:   2,
            chess.QUEEN:  4,
            chess.KING:   0
        }
        max_phase = sum(phase_weights[p] * 2 for p in phase_weights)  # both sides

        # --- 3) PSTs for middle-game and endgame (64‐length lists) ---
        #   (you can tweak these values or replace with more complete tables)
        mg_pst = {
            chess.PAWN:   [0]*64,  # simple placeholder
            chess.KNIGHT: [
                -50,-40,-30,-30,-30,-30,-40,-50,
                -40,-20,  0,  5,  5,  0,-20,-40,
                -30,  5, 10, 15, 15, 10,  5,-30,
                -30,  0, 15, 20, 20, 15,  0,-30,
                -30,  5, 15, 20, 20, 15,  5,-30,
                -30,  0, 10, 15, 15, 10,  0,-30,
                -40,-20,  0,  0,  0,  0,-20,-40,
                -50,-40,-30,-30,-30,-30,-40,-50
            ],
            chess.BISHOP: [
                -20,-10,-10,-10,-10,-10,-10,-20,
                -10,  5,  0,  0,  0,  0,  5,-10,
                -10, 10, 10, 10, 10, 10, 10,-10,
                -10,  0, 10, 10, 10, 10,  0,-10,
                -10,  5,  5, 10, 10,  5,  5,-10,
                -10,  0,  5, 10, 10,  5,  0,-10,
                -10,  0,  0,  0,  0,  0,  0,-10,
                -20,-10,-10,-10,-10,-10,-10,-20
            ],
            chess.ROOK: [
                  0,  0,  5, 10, 10,  5,  0,  0,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                 -5,  0,  0,  0,  0,  0,  0, -5,
                  5, 10, 10, 10, 10, 10, 10,  5,
                  0,  0,  0,  0,  0,  0,  0,  0
            ],
            chess.QUEEN: [0]*64,
            chess.KING: [
               20, 30, 10,  0,  0, 10, 30, 20,
               20, 20,  0,  0,  0,  0, 20, 20,
              -10,-20,-20,-20,-20,-20,-20,-10,
              -20,-30,-30,-40,-40,-30,-30,-20,
              -30,-40,-40,-50,-50,-40,-40,-30,
              -30,-40,-40,-50,-50,-40,-40,-30,
              -30,-40,-40,-50,-50,-40,-40,-30,
              -30,-40,-40,-50,-50,-40,-40,-30
            ]
        }
        eg_pst = {
            # In endgame we often prefer king more central; simple example:
            **mg_pst,
            chess.KING: [
                 -50,-40,-30,-20,-20,-30,-40,-50,
                 -30,-20,-10,  0,  0,-10,-20,-30,
                 -30,-10, 20, 30, 30, 20,-10,-30,
                 -30,-10, 30, 40, 40, 30,-10,-30,
                 -30,-10, 30, 40, 40, 30,-10,-30,
                 -30,-10, 20, 30, 30, 20,-10,-30,
                 -30,-30,  0,  0,  0,  0,-30,-30,
                 -50,-30,-30,-30,-30,-30,-30,-50
            ]
        }

        # --- 4) Compute game-phase factor ---
        phase = 0
        for (color, ptype), bb in bitboards.items():
            cnt = bitboard.popcount(bb)
            phase += phase_weights[ptype] * cnt
        phase = max(0, min(phase, max_phase))
        mg_phase = phase / max_phase
        eg_phase = 1.0 - mg_phase

        # --- 5) Accumulate material + PSTs ---
        score = 0.0
        for (color, ptype), bb in bitboards.items():
            pcs = bitboard.popcount(bb)
            mat = mat_values[ptype] * pcs
            # for each bit in bb, add PST contribution
            pst_score = 0
            b = bb
            while b:
                sq = (b & -b).bit_length() - 1
                b &= b - 1
                # use mg and eg PST, mirror black squares
                mg_val = mg_pst[ptype][sq if color=='white' else sq^56]
                eg_val = eg_pst[ptype][sq if color=='white' else sq^56]
                pst_score += mg_phase * mg_val + eg_phase * eg_val

            total = mat + pst_score
            score += total if color=='white' else -total

        return score if ai_color=='white' else -score

    def _fen_to_tensor(self, fen_str: str):
        """
        Wraps your existing fen_to_features and tensor conversion.
        """
        from nnue.nnue_train import fen_to_features
        arr = fen_to_features(fen_str)
        return torch.from_numpy(arr).unsqueeze(0)