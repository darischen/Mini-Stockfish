# ai.py
import math
import os
import time
import torch  # Ensure PyTorch is installed if you plan to use a DNN
import torch.nn as nn
import chess  # Use python-chess for fast bitboard-backed move generation
import numpy as np
import bitboard  # Our bitboard module (square_bb, popcount, attacks, etc.)
from nnue.nnue_train import NNUEModel
from move import Move  # Move class for interoperability with the game engine
from square import Square
from accumulator import Accumulator  # Accumulator for incremental feature updates
import threading
from chess.polyglot import zobrist_hash
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class TranspositionTable:
    def __init__(self):
        self.lock = threading.Lock()
        self.table = {}  # key -> {'depth': int, 'value': float}

    def get(self, key, depth):
        with self.lock:
            entry = self.table.get(key)
            if entry and entry['depth'] >= depth:
                return entry['value']
            return None

    def store(self, key, depth, value):
        with self.lock:
            # Always overwrite or insert
            self.table[key] = {'depth': depth, 'value': value}
            
class ChessAI:
    def __init__(self, depth=3, use_dnn=False, model_path=None):
        """
        Initialize the chess AI.
        :param depth: How many plies to search.
        :param use_dnn: Whether to use a deep neural network (NNUE) for evaluation.
        :param model_path: Path to the pretrained model.
        """
        self.tt = TranspositionTable()
        self.stats_lock = threading.Lock()
        self.depth = depth
        self.use_dnn = use_dnn
        
        if self.use_dnn and model_path and os.path.isfile(model_path):
            # Load the compiled TorchScript model on CPU
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            # Optimize CPU threading for small models
            torch.set_num_threads(os.cpu_count() or 1)
        else:
            self.model = None

    def choose_move(self, board, color: str):
        """
        Iterative deepening with one tqdm per depth.
        """
        # reset overall stats
        self.nodes_evaluated = 0
        self.branches_pruned = 0

        root_fen   = board.get_fen()
        root_board = chess.Board(root_fen)
        # set the side to move correctly
        root_board.turn = chess.WHITE if color == 'white' else chess.BLACK

        best_move = None
        best_eval = -math.inf if color == 'white' else math.inf

        total_start = time.time()

        # iterate depths 1..self.depth
        for depth in range(1, self.depth + 1):
            bar = tqdm(desc=f"Depth {depth}", total=None)
            maximizing = (color == 'white')
            current_best = None
            current_eval = -math.inf if maximizing else math.inf

            moves = list(root_board.legal_moves)
            # launch one thread per root move
            with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
                futures = [
                    executor.submit(
                        self._search_move,
                        root_fen,
                        uci,
                        depth,
                        maximizing,
                        color,
                        bar
                    )
                    for uci in moves
                ]
                # collect results
                for fut in as_completed(futures):
                    val, uci = fut.result()
                    if maximizing:
                        if val > current_eval:
                            current_eval, current_best = val, uci
                    else:
                        if val < current_eval:
                            current_eval, current_best = val, uci

            bar.close()
            best_move, best_eval = current_best, current_eval
            print(f"Depth {depth} → best={best_move} eval={best_eval:.4f}")

        elapsed = time.time() - total_start
        print(f"AI search complete. Nodes: {self.nodes_evaluated}, Pruned: {self.branches_pruned}, Time: {elapsed:.2f}s")
        print(f"Best eval for {color}: {best_eval:.4f}")

        if best_move is None:
            return None

        # map UCI back to your Move/Square classes
        src, dst = best_move.from_square, best_move.to_square
        sr, sf = divmod(src, 8)
        dr, df = divmod(dst, 8)
        initial = Square(7 - sr, sf)
        final   = Square(7 - dr, df)
        mv = Move(initial, final)
        piece = board.squares[initial.row][initial.col].piece
        return (piece, mv)
    
    def _search_move(self, root_fen, uci, depth, maximizing, ai_color, progress_bar):
        # Each thread gets its own board and accumulator
        board = chess.Board(root_fen)
        board.turn = chess.BLACK
        acc = Accumulator()
        acc.init(board)

        captured = board.piece_at(uci.to_square)
        board.push(uci)
        acc.update(uci, captured)
        val = self._minimax(
            board, acc,
            depth - 1,
            -math.inf, math.inf,
            not maximizing,
            ai_color,
            progress_bar
        )
        return val, uci

    def _minimax(self,
                 board: chess.Board,
                 acc: Accumulator,
                 depth: int,
                 alpha: float,
                 beta: float,
                 maximizing_player: bool,
                 ai_color: str,
                 progress_bar: tqdm):
        """
        Alpha‐beta + TT + accumulator + per‐node progress updates.
        """
        # stats + bar update
        with self.stats_lock:
            self.nodes_evaluated += 1
            progress_bar.update(1)
            progress_bar.set_postfix({
                "nodes":  self.nodes_evaluated,
                "pruned": self.branches_pruned
            })

        key = zobrist_hash(board)
        if (cached := self.tt.get(key, depth)) is not None:
            return cached

        # leaf
        if depth == 0 or board.is_game_over():
            val = self.model(acc.state).item() if self.model else self._evaluate_bb(board, ai_color)
            self.tt.store(key, depth, val)
            return val

        if maximizing_player:
            value = -math.inf
            for mv in self._order_moves(board, True):
                cap = board.piece_at(mv.to_square)
                board.push(mv); acc.update(mv, cap)
                child = self._minimax(board, acc, depth - 1, alpha, beta, False, ai_color, progress_bar)
                acc.rollback(mv, cap); board.pop()

                value = max(value, child)
                alpha = max(alpha, child)
                if beta <= alpha:
                    with self.stats_lock:
                        self.branches_pruned += 1
                        progress_bar.set_postfix(pruned=self.branches_pruned)
                    break

            self.tt.store(key, depth, value)
            return value

        else:
            value = math.inf
            for mv in self._order_moves(board, False):
                cap = board.piece_at(mv.to_square)
                board.push(mv); acc.update(mv, cap)
                child = self._minimax(board, acc, depth - 1, alpha, beta, True, ai_color, progress_bar)
                acc.rollback(mv, cap); board.pop()

                value = min(value, child)
                beta = min(beta, child)
                if beta <= alpha:
                    with self.stats_lock:
                        self.branches_pruned += 1
                        progress_bar.set_postfix(pruned=self.branches_pruned)
                    break
            self.tt.store(key, depth, value)
            return value

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
            chess.PAWN: [
                 0,   0,   0,   0,   0,   0,   0,   0,
                78,  83,  86,  73, 102,  82,  85,  90,
                 7,  29,  21,  44,  40,  31,  44,   7,
               -17,  16,  -2,  15,  14,   0,  15, -13,
               -26,   3,  10,   9,   6,   1,   0, -23,
               -22,   9,   5, -11, -10,  -2,   3, -19,
               -31,   8,  -7, -37, -36, -14,   3, -31,
                 0,   0,   0,   0,   0,   0,   0,   0
            ],
            chess.KNIGHT: [
               -66, -53, -75, -75, -10, -55, -58, -70,
                -3,  -6, 100, -36,   4,  62,  -4, -14,
                10,  67,   1,  74,  73,  27,  62,  -2,
                24,  24,  45,  37,  33,  41,  25,  17,
                -1,   5,  31,  21,  22,  35,   2,   0,
               -18,  10,  13,  22,  18,  15,  11, -14,
               -23, -15,   2,   0,   2,   0, -23, -20,
               -74, -23, -26, -24, -19, -35, -22, -69
            ],
            chess.BISHOP: [
               -59, -78, -82, -76, -23,-107, -37, -50,
               -11,  20,  35, -42, -39,  31,   2, -22,
                -9,  39, -32,  41,  52, -10,  28, -14,
                25,  17,  20,  34,  26,  25,  15,  10,
                13,  10,  17,  23,  17,  16,   0,   7,
                14,  25,  24,  15,   8,  25,  20,  15,
                19,  20,  11,   6,   7,   6,  20,  16,
                -7,   2, -15, -12, -14, -15, -10, -10
            ],
            chess.ROOK: [
                35,  29,  33,   4,  37,  33,  56,  50,
                55,  29,  56,  67,  55,  62,  34,  60,
                19,  35,  28,  33,  45,  27,  25,  15,
                 0,   5,  16,  13,  18,  15,   2,   0,
                -2,   0,   2,   0,   0,   0,  -2, -15,
               -14, -14, -10, -10,  -2, -10, -14, -14,
               -14, -17, -18, -19, -19, -16, -17, -14,
               -23, -23, -23, -23, -23, -23, -23, -23
            ],
            chess.QUEEN: [
               -28,   0,  29,  12,  59,  44,  43,  45,
               -24, -39,  -5,   1, -16,  57,  28,  54,
               -13, -17,   7,   8,  29,  56,  47,  57,
               -27, -27, -16, -16,  -1,  17,  -2,  -9,
               -23,  -9,  12,  10,  19,  17,  -2,  -5,
               -22, -17,   4,   3,  14,   5,  -5, -17,
               -31, -28, -19, -21, -15, -22, -28, -31,
               -36, -53, -60, -64, -64, -60, -53, -36
            ],
            chess.KING: [
               -65,  23,  16, -15, -56, -34,   2,  13,
                29,  -1, -20,  -7,  -8,  -4, -38, -29,
                -9,  24,   2, -16, -20,   6,  22, -22,
               -17, -20, -12, -27, -30, -25, -14, -36,
               -49,  -1, -27, -39, -46, -44, -33, -51,
               -14, -14, -22, -46, -44, -30, -15, -27,
                 1,   7,  13, -13, -16,   3,   7, -35,
                40,  41,  26,  21,  16,  27,  47,  57
            ]
        }
        eg_pst = {
            chess.PAWN: [
                 0,   0,   0,   0,   0,   0,   0,   0,
                178, 173, 158, 134, 147, 132, 165, 187,
                 94, 100,  85,  67,  56,  53,  82,  84,
                 32,  24,  13,   5,  -2,   4,  -6,  -8,
                 -6,   9,  -9,  -5,  -1,  -3,   0,  -7,
                 -8,   2,  -6,  -2,  -2,   1,  -1,  -8,
                -16,   0, -12,  -7,  -4,  -6,   3, -14,
                  0,   0,   0,   0,   0,   0,   0,   0
            ],
            chess.KNIGHT: [
               -58, -38, -13, -28, -31, -27, -63, -99,
               -25,  -8, -25,  -2,  -9, -25, -24, -52,
                -24, -20,  10,   9,  -1,  -9, -19, -41,
                -17,   3,  22,  22,  22,   8,   0, -20,
                -18, -11,  11,  12,  12,   7,   4, -17,
                -23,  -9,   1,   11,  10,   7,  -9, -23,
                -29, -53, -12,  -3,  -1, -32, -55, -58,
                -64, -31, -35, -13, -24, -19, -29, -78
            ],
            chess.BISHOP: [
               -14, -21, -11, -8, -7, -9, -17, -24,
                -8,  -4,   7, -12, -3, -13, -4, -14,
                 2,  -1,  -8, -10, -14, -15,  -2,   5,
                -3,   9,   12,   9,   14,   10,    3,   -1,
                -6,    3,   13,   19,   7,   10,   -3,   -9,
               -12,   -3,    8,   10,   13,    3,   -7,  -15,
               -14,  -18,   -7,   -1,    4,   -9,  -15,  -27,
               -23,   -9,  -23,   -5,   -9,  -16,   -5,  -17
            ],
            chess.ROOK: [
                13,   10,   18,   15,   12,   12,    8,    5,
                11,   13,   13,   11,    -3,    3,    8,    3,
                 7,    7,    7,    5,     4,    5,    3,    5,
                 3,    5,    5,    4,     5,    5,    4,    3,
                 2,    2,    2,    2,     3,    2,    2,    2,
                 1,    1,    2,    2,     2,    2,    1,    1,
                 2,    2,    2,    2,     2,    2,    2,    2,
                 2,    2,    2,    2,     2,    2,    2,    2
            ],
            chess.QUEEN: [
                -9,    3,   -5,   -1,   -5,  -13,    4,  -20,
                -3,   -9,   10,   -5,    3,    6,   -6,  -12,
                -6,   -1,    9,   10,  -23,  -11,   -3,   -9,
                -3,   -1,   -5,  -13,  -15,  -14,    2,  -22,
                -9,  -18,  -10,   -8,   -8,  -15,  -10,  -23,
               -16,  -27,   -9,   -5,   -9,  -23,  -11,  -23,
               -22,  -23,  -16,  -21,  -13,  -29,  -23,  -31,
               -55,  -38,  -19,  -29,  -53,  -38,  -37,  -64
            ],
            chess.KING: [
               -74,  -35,  -18,  -18,  -11,  -19,  -23,  -28,
               -33,  -11,    4,    9,     6,    4,  -11,  -16,
               -45,  -16,   37,   42,    42,   37,  -16,  -58,
               -74,  -44,   -2,   -1,    -1,   -2,  -44,  -71,
               -36,  -26,  -12,   -1,     2,  -12,  -24,  -43,
               -30,  -16,  -21,   -6,    -8,  -16,  -15,  -28,
               -31,  -19,  -18,  -10,    -7,  -11,  -17,  -30,
               -49,  -32,  -31,  -29,   -30,  -31,  -36,  -50
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
        
        final_score = score if ai_color =='white' else -score
        # --- Bonus for Checks ---
        if bb.is_check():
            check_bonus = 50  # increase/decrease as necessary
            if bb.turn == (ai_color == 'white'):
                final_score += check_bonus
            else:
                final_score -= check_bonus

        # --- Bonus for Promotions ---
        promotion_bonus = 200  # strong bonus to prioritize promotions
        for move in bb.legal_moves:
            if move.promotion:
                if bb.turn == (ai_color == 'white'):
                    final_score += promotion_bonus
                else:
                    final_score -= promotion_bonus

        return score if ai_color=='white' else -score

    def _fen_to_tensor(self, fen_str: str):
        """
        Wraps your existing fen_to_features and tensor conversion.
        """
        from nnue.nnue_train import fen_to_features
        arr = fen_to_features(fen_str)
        return torch.from_numpy(arr).unsqueeze(0)