# ai.py
import math
import os, threading
import time
import torch  # Ensure PyTorch is installed if you plan to use a DNN
import chess  # Use python-chess for fast bitboard-backed move generation
import bitboard  # Our bitboard module (square_bb, popcount, attacks, etc.)
from move import Move  # Move class for interoperability with the game engine
from square import Square
from accumulator import Accumulator  # Accumulator for incremental feature updates
from chess.polyglot import zobrist_hash
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from chess import SquareSet
import json
import chess
from chess.syzygy import Tablebase as SyzygyTablebase
from chess.gaviota import PythonTablebase as GaviotaTablebase
from core_search import minimax
import core_search

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
syzygy_tb = SyzygyTablebase()
syzygy_tb.add_directory("endgame/syzygy/")

gaviota_tb = GaviotaTablebase()
gaviota_tb.add_directory("endgame/gaviota/3/")
gaviota_tb.add_directory("endgame/gaviota/4/")
gaviota_tb.add_directory("endgame/gaviota/5/")

from chess import KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN
piece_map = {
    PAWN:   'P',
    KNIGHT: 'N',
    BISHOP: 'B',
    ROOK:   'R',
    QUEEN:  'Q',
    KING:   'K',
}

EXACT      = 0
LOWERBOUND = 1
UPPERBOUND = 2
            
class ChessAI:
    PIECE_VALUES = (0, 100, 300, 310, 400, 900, 20000)
    def __init__(self, depth=3, use_dnn=False, model_path=None):
        """
        Initialize the chess AI.
        :param depth: How many plies to search.
        :param use_dnn: Whether to use a deep neural network (NNUE) for evaluation.
        :param model_path: Path to the pretrained model.
        """
        self.stats_lock = threading.Lock()
        self.depth = depth
        self.use_dnn = use_dnn
        core_search.set_use_nnue(self.use_dnn)
        core_search.init_nnue("nnue/64indepth_int8.pt") # "nnue/hidden64best1.057e-2_int8.pt"
        
        # Book moves are stored in book/book.json
        with open("book/book.json") as f:
            data = json.load(f)
        self.book_evals = {int(k): v for k, v in data.items()}
        print(f"[DEBUG] loaded book.json: {len(self.book_evals)} entries")

        # how many plies deep your opening book should go
        self.book_depth = 20
        
        if self.use_dnn and model_path and os.path.isfile(model_path):
            # Load the compiled TorchScript model on CPU
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            # Optimize CPU threading for small models
            torch.set_num_threads(os.cpu_count() or 1)
        else:
            self.model = None
    
    def _uci_to_move(self, board: chess.Board, uci_move: chess.Move):
        src, dst = uci_move.from_square, uci_move.to_square
        sr, sf = divmod(src, 8)
        dr, df = divmod(dst, 8)
        initial = Square(7 - sr, sf)
        final   = Square(7 - dr, df)
        mv = Move(initial, final)
        mv.initial = initial
        mv.final   = final
        return mv
    
    def choose_move(self, board, color: str):
        """
        Iterative deepening with per-depth tqdm, parallel root evaluation using Cython minimax,
        and per-node bar updates matching the original Python version.
        """
        # —— SPECIAL-CASE: mate in 1 ——
        root_board = chess.Board(board.get_fen())
        root_board.turn = chess.WHITE if color=='white' else chess.BLACK
        for uci in root_board.legal_moves:
            root_board.push(uci)
            if root_board.is_checkmate():
                mv = self._uci_to_move(board, uci)
                piece = board.squares[mv.initial.row][mv.initial.col].piece
                return piece, mv
            root_board.pop()
            
        # ——— Endgame tablebase shortcut for ≤5 pieces ———
        if len(root_board.piece_map()) <= 5:
            win_moves, draw_moves, loss_moves = [], [], []

            for uci in root_board.legal_moves:
                root_board.push(uci)

                # 1) Depth‐to‐mate from Gaviota
                dtm = gaviota_tb.probe_dtm(root_board)
                # 2) Win/Draw/Loss from Syzygy
                raw = syzygy_tb.probe_wdl(root_board)
                wdl = -raw
                if wdl == 2:          # win for side to move
                    win_moves.append((dtm, uci))
                elif wdl == -2:                 # loss
                    loss_moves.append((dtm, uci))
                else:                           # draw or cursed win/loss
                    dtz = syzygy_tb.probe_dtz(root_board) or float('inf')
                    draw_moves.append((dtz, uci))
                root_board.pop()

            # pick fastest win, else quickest draw, else slowest loss
            if   win_moves:
                _, best_uci = max(win_moves, key=lambda x: x[0])
            elif draw_moves:
                _, best_uci = min(draw_moves, key=lambda x: x[0])
            elif loss_moves:
                _, best_uci = max(loss_moves, key=lambda x: x[0])
            else:
                best_uci = None

            if best_uci is not None:
                mv = self._uci_to_move(board, best_uci)
                piece = board.squares[mv.initial.row][mv.initial.col].piece
                return piece, mv
                    
        # Book Moves
        root_board = chess.Board(board.get_fen())
        root_board.turn = chess.WHITE if color=='white' else chess.BLACK
        ply = (root_board.fullmove_number - 1) * 2 \
              + (0 if root_board.turn == chess.WHITE else 1)

        key = zobrist_hash(root_board)
        print("  fen:", root_board.fen())
        print("  in_book?", key in self.book_evals)
        
        print(f"[DEBUG] ply={ply}, root_key={key}, in_book={ key in self.book_evals }")
        if ply < self.book_depth and key in self.book_evals:
            best_move, best_score = None, (math.inf if color=='black' else -math.inf)
            # try every legal move, pick the child whose hash is in book_evals
            for move in root_board.legal_moves:
                root_board.push(move)
                child_key = zobrist_hash(root_board)
                root_board.pop()

                # only consider it if we actually precomputed it
                if child_key in self.book_evals:
                    score = self.book_evals[child_key]
                    if color == 'black':
                        if score < best_score:
                            best_score, best_move = score, move
                    else:
                        if score > best_score:
                            best_score, best_move = score, move

            if best_move is not None:
                mv = self._uci_to_move(root_board, best_move)
                piece = board.squares[mv.initial.row][mv.initial.col].piece
                return piece, mv
        
        # Main Search
        
        core_search.reset_counters()
        
        core_search.set_use_nnue(self.use_dnn)

        root_fen = board.get_fen()
        maximize = (color == 'white')
        best_move = None
        best_eval = -math.inf if maximize else math.inf
        
        max_workers = os.cpu_count() or 1
        executor = ThreadPoolExecutor(max_workers=max_workers)
        executor.submit(lambda: None).result()

        total_start = time.time()

        # iterate depths 1..self.depth
        for depth in range(1, self.depth + 1):
            core_search.reset_counters()
            bar = tqdm(desc=f"Depth {depth}", total=None)

            # snapshot of nodes before this depth
            nodes_before = core_search.get_nodes_evaluated()

            # launch parallel root evaluations
            root_board = chess.Board(root_fen)
            root_board.turn = chess.WHITE if color == 'white' else chess.BLACK
            moves = list(root_board.legal_moves)
            futures = [
                executor.submit(
                    self._evaluate_root,
                    root_fen,
                    uci,
                    depth,
                    maximize,
                    color
                ) for uci in moves
            ]
            current_best = None
            current_eval = -math.inf if maximize else math.inf
            for fut in as_completed(futures):
                val, uci = fut.result()
                # update progress by number of nodes this branch consumed
                nodes_after = core_search.get_nodes_evaluated()
                delta = nodes_after - nodes_before
                nodes_before = nodes_after
                bar.update(delta)
                bar.set_postfix({
                    'nodes': nodes_after,
                    'pruned': core_search.get_branches_pruned()
                })

                if maximize:
                    if val > current_eval:
                        current_eval, current_best = val, uci
                else:
                    if val < current_eval:
                        current_eval, current_best = val, uci

            bar.close()
            best_move, best_eval = current_best, current_eval
            print(f"Depth {depth} → best={best_move} eval={best_eval:.4f}")

        elapsed = time.time() - total_start
        print(f"AI search complete. Nodes: {core_search.get_nodes_evaluated()}, Pruned: {core_search.get_branches_pruned()}, Time: {elapsed:.2f}s")
        print(f"Best eval for {color}: {best_eval:.4f}")
        executor.shutdown()

        if best_move is None:
            return None

        # map UCI back to your Move/Square classes
        mv = self._uci_to_move(board, best_move)
        piece = board.squares[mv.initial.row][mv.initial.col].piece
        return piece, mv

    def _evaluate_root(self, root_fen, uci, depth, maximize, ai_color):
        """Evaluate one root move via Cython minimax."""
        board = chess.Board(root_fen)
        board.turn = chess.WHITE if ai_color == 'white' else chess.BLACK
        acc = Accumulator(); acc.init(board)

        captured = board.piece_at(uci.to_square)
        board.push(uci); acc.update(uci, captured)

        root_key = zobrist_hash(board)
        val = minimax(board, acc,
                depth - 1,
                -math.inf, math.inf,
                ai_color,
                root_key,
                depth)
        return val, uci
    
    def _stand_pat(self,
                   acc: Accumulator,
                   board: chess.Board,
                   ai_color: str) -> float:
        """Compute blended NNUE/static eval, then apply all handcrafted bonuses/penalties."""
        if self.model:
           base = self.model(acc.state).item()
        else:
           base = self._evaluate_bb(board, ai_color)
        
        adj  = base + self.bonus(board, base, ai_color)
        return adj if ai_color=='white' else -adj

    def bonus(self,
                                     board: chess.Board,
                                     base_score: float,
                                     ai_color: str) -> float:
        """Tack on every hanging‑piece penalty, queen safety, pawn‑attack bonus, etc."""
        s = base_score
        us = chess.WHITE if ai_color=='white' else chess.BLACK
        them = not us

        # — hanging‑piece penalty
        pen = 1000
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc and pc.color==us:
                if board.is_attacked_by(them, sq) and not board.is_attacked_by(us, sq):
                    s -= pen if board.turn==us else -pen

        # — bonus for unprotected enemy captures
        bonus = 300
        for m in board.legal_moves:
            if board.is_capture(m):
                vic = board.piece_at(m.to_square)
                if vic and not board.is_attacked_by(us, m.to_square):
                    s += (1 if board.turn==us else -1)*bonus

        # — mobility bonus
        mob = len(list(board.legal_moves))
        s += (1 if board.turn==us else -1)*mob*10

        # — bishop‑pair
        if sum(1 for p in board.piece_map().values()
               if p.piece_type==chess.BISHOP and p.color==us)==2:
            s += 50

        # — check bonus
        if board.is_check():
            s += (100 if board.turn==us else -100)

        # — promotion bonus
        for m in board.legal_moves:
            if m.promotion:
                s += (500 if board.turn==us else -500)

        # — queen safety & mobility
        qa, qs, qm = 500, 75, 20
        for sq, p in board.piece_map().items():
            if p.piece_type==chess.QUEEN and p.color==us:
                attacked = board.is_attacked_by(them, sq)
                defended = board.is_attacked_by(us, sq)
                if attacked and not defended: s -= qa
                rank = chess.square_rank(sq)
                if (us==chess.WHITE and rank<=2) or (us==chess.BLACK and rank>=5):
                    s += qs
                s += sum(1 for m in board.legal_moves if m.from_square==sq)*qm
                break

        # — pawn‑attack bonus
        pawn_bonus = {chess.KNIGHT:50,
                      chess.BISHOP:50,
                      chess.ROOK:75,
                      chess.QUEEN:100}
        for sq, p in board.piece_map().items():
            if p.piece_type == chess.PAWN and p.color == us:
                mask = board.attacks_mask(sq)
                for tgt in chess.SquareSet(mask):
                    vic = board.piece_at(tgt)
                    if vic and vic.color != us and vic.piece_type in pawn_bonus:
                        s += pawn_bonus[vic.piece_type]
                        
        # - capture checking piece bonus
        if board.is_check():
            us = chess.WHITE if ai_color=='white' else chess.BLACK
            # bitboard of all checkers
            checkers_bb = board.checkers()
            for checker_sq in SquareSet(checkers_bb):
                # if you have a legal move that lands on the checker’s square
                for m in board.legal_moves:
                    if m.to_square == checker_sq:
                        s += 1000  # tweak this as needed
                        # once per checker is enough
                        break
        
        # — piece safety bonus
        W = 50.0
        for sq, pc in board.piece_map().items():
            if pc.color == us:
                attackers = board.attackers(them, sq)
                defenders = board.attackers(us,   sq)
                net_threat = len(attackers) - len(defenders)
                if net_threat > 0:
                    # PIECE_VALUES maps pawn→100, knight→300, …, queen→900
                    s -= W * net_threat * ChessAI.PIECE_VALUES[pc.piece_type]
                
        return s
    
    def _evaluate_bb(self, board: chess.Board, ai_color: str):
        """
        Bitboard-based static evaluation using PST interpolation.
        """
        # --- 1) Build per-piece bitboards from python-chess ---
        bitboards = {}
        for sq, pc in board.piece_map().items():
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
        for (color, ptype), mask in bitboards.items():
            cnt = bitboard.popcount(mask)
            phase += phase_weights[ptype] * cnt
        phase = max(0, min(phase, max_phase))
        mg_phase = phase / max_phase
        eg_phase = 1.0 - mg_phase

        # --- 5) Accumulate material + PSTs ---
        score = 0.0
        for (color, ptype), mask in bitboards.items():
            pcs = bitboard.popcount(mask)
            mat = mat_values[ptype] * pcs
            # for each bit in bb, add PST contribution
            pst_score = 0
            b = mask
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
        
        # --- Penalty for Hanging a Piece ---
        own_penalty = 200
        # scan every square on the board
        for sq in chess.SQUARES:
            pc = board.piece_at(sq)
            if pc and pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK):
                # attacked by opponent but not defended by you
                if board.is_attacked_by(not board.turn, sq) and not board.is_attacked_by(board.turn, sq):
                    final_score -= own_penalty
                    
        # --- Bonus for hanging enemy captures ---
        hanging_bonus = 150
        for mv in board.legal_moves:
            if board.is_capture(mv):
                vic = board.piece_at(mv.to_square)
                # if the victim is unprotected on its square
                if vic and not board.is_attacked_by(board.turn, mv.to_square):
                    # board.turn is side to move who would do this capture
                    sign = 1 if (board.turn == (ai_color == 'white')) else -1
                    final_score += sign * hanging_bonus
        
        # --- Open Position Bonus ---
        mobility = board.legal_moves.count()
        mob_bonus = 10
        # if it's your turn, reward your mobility, else penalize
        if board.turn == (ai_color == 'white'):
            final_score += mobility * mob_bonus
        else:
            final_score -= mobility * mob_bonus
            
        # --- Bishop‑pair bonus ---
        # give a small bonus if you still have both bishops
        bishops = 0
        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.BISHOP and pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK):
                bishops += 1
        if bishops == 2:
            pair_bonus = 50
            final_score += pair_bonus
                    
        # --- Bonus for Checks ---
        if board.is_check():
            check_bonus = 100
            if board.turn == (ai_color == 'black'):
                final_score += check_bonus
            else:
                final_score -= check_bonus

        # --- Bonus for Promotions ---
        promotion_bonus = 200
        for move in board.legal_moves:
            if move.promotion:
                if board.turn == (ai_color == 'white'):
                    final_score += promotion_bonus
                else:
                    final_score -= promotion_bonus
                    
        # --- Queen Safety & Mobility ---
        queen_attack_penalty = 500
        queen_safe_bonus    = 75
        queen_mobility_bonus = 20

        # scan for your queen
        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.QUEEN and pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK):
                # 1) attacked & under‑defended?
                attacked   = board.is_attacked_by(not board.turn, sq)
                defended   = board.is_attacked_by(board.turn, sq)
                sign = 1 if pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK) else -1
                if attacked and not defended:
                    final_score -= queen_attack_penalty * sign

                # 2) queen safely behind pawn shield?
                rank = chess.square_rank(sq)  # 0=rank1, …,7=rank8
                if pc.color == chess.WHITE:
                    # safe if on ranks 1–3 (behind pawns)
                    if rank <= 2:
                        final_score += queen_safe_bonus * sign
                else:
                    # black safe on ranks 6–8
                    if rank >= 5:
                        final_score += queen_safe_bonus * sign

                # 3) queen mobility
                qm = 0
                for m in board.legal_moves:
                    if m.from_square == sq:
                        qm += 1
                final_score += qm * queen_mobility_bonus * sign
        
        # --- Pawn‐attack on higher‐value piece bonus ---
        pawn_attack_bonus = {
            chess.KNIGHT:  50,
            chess.BISHOP:  50,
            chess.ROOK:    75,
            chess.QUEEN:  100,
        }
        # for each pawn of the AI’s side
        for sq, pc in board.piece_map().items():
            if pc.piece_type == chess.PAWN and pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK):
                # squares this pawn could attack
                mask = board.attacks_mask(sq)
                for tgt in chess.SquareSet(mask):
                    vic = board.piece_at(tgt)
                    # if there’s an enemy piece of greater value
                    if vic and vic.color != pc.color and vic.piece_type in pawn_attack_bonus:
                        sign = 1 if pc.color == (chess.WHITE if ai_color=='white' else chess.BLACK) else -1
                        final_score += sign * pawn_attack_bonus[vic.piece_type]

        return final_score if ai_color=='white' else -final_score

    def _fen_to_tensor(self, fen_str: str):
        """
        Wraps your existing fen_to_features and tensor conversion.
        """
        from nnue.nnue_train import fen_to_features
        arr = fen_to_features(fen_str)
        return torch.from_numpy(arr).unsqueeze(0)