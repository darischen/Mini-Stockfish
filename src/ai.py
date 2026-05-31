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
from concurrent.futures import ThreadPoolExecutor, as_completed  # kept for potential future use
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
try:
    syzygy_tb.add_directory("endgame/syzygy/")
except (FileNotFoundError, OSError):
    pass  # Tablebase directory not found, continue without it

gaviota_tb = GaviotaTablebase()
try:
    gaviota_tb.add_directory("endgame/gaviota/3/")
    gaviota_tb.add_directory("endgame/gaviota/4/")
    gaviota_tb.add_directory("endgame/gaviota/5/")
except (FileNotFoundError, OSError):
    pass  # Tablebase directory not found, continue without it

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
        core_search.init_nnue("nnue/halfkp_int8.pt")
        
        # Book moves are stored in book/book.json
        with open("book/book.json") as f:
            data = json.load(f)
        self.book_evals = {int(k): v for k, v in data.items()}
        print(f"[DEBUG] loaded book.json: {len(self.book_evals)} entries")

        # how many plies deep your opening book should go
        self.book_depth = 20

        # Track when we leave the opening book to avoid repeated lookups
        self.out_of_book = False

        if self.use_dnn and model_path and os.path.isfile(model_path):
            # Load the compiled TorchScript model on CPU
            self.model = torch.jit.load(model_path, map_location="cpu")
            self.model.eval()
            # Optimize CPU threading for small models
            torch.set_num_threads(os.cpu_count() or 1)
        else:
            self.model = None

    def reset(self):
        """Reset AI state for a new game."""
        self.out_of_book = False

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

    def _is_blunder(self, board: chess.Board, move: chess.Move) -> bool:
        """
        Check if a move is a blunder by putting a valuable piece on an attacked square
        without adequate defense. Returns True if the move is likely a blunder.
        """
        piece = board.piece_at(move.from_square)
        if not piece:
            return False

        # Get piece value
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
                        chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        our_value = piece_values.get(piece.piece_type, 0)

        # Only check for valuable pieces (knight and above)
        if our_value < 3:
            return False

        us = piece.color
        them = not us

        # Check if destination is attacked by enemy
        if not board.is_attacked_by(them, move.to_square):
            return False  # Safe destination

        # Check if we have adequate defense
        # IMPORTANT: Exclude the moving piece itself from defenders - it can't defend its destination
        defenders = len([sq for sq in board.attackers(us, move.to_square)
                         if sq != move.from_square])
        attackers = len(list(board.attackers(them, move.to_square)))

        # If a capture, check if we win material
        victim = board.piece_at(move.to_square)
        if victim:
            victim_value = piece_values.get(victim.piece_type, 0)
            # If we capture something of equal or greater value, might be ok
            if victim_value >= our_value:
                return False
            # Even if victim is less valuable, consider if we're defended
            if defenders > 0 and victim_value >= our_value - 2:
                return False

        # Queen moving to attacked square with fewer defenders than attackers = blunder
        if piece.piece_type == chess.QUEEN:
            if attackers > defenders:
                return True

        # Hanging piece (attacked, not defended)
        if defenders == 0:
            return True

        # Piece moving to square where lowest attacker < our value
        # Simple check: if attacked by pawn and we're not a pawn, bad
        pawn_attackers = [sq for sq in board.attackers(them, move.to_square)
                         if board.piece_at(sq) and board.piece_at(sq).piece_type == chess.PAWN]
        if pawn_attackers and our_value > 1:
            return True

        return False

    def _filter_blunders(self, board: chess.Board, moves: list) -> list:
        """Filter out obvious blunder moves, keeping at least one move."""
        safe_moves = [m for m in moves if not self._is_blunder(board, m)]
        # Always keep at least some moves to avoid returning nothing
        return safe_moves if safe_moves else moves

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
                    
        # Book Moves - skip if we've already left the book
        if not self.out_of_book:
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

            # No book move found - mark as out of book
            self.out_of_book = True
            print("[DEBUG] Left opening book")

        # Main Search

        core_search.reset_counters()
        core_search.reset_tt_counters()
        core_search.clear_tt()  # fresh TT for this move's search
        core_search.clear_history()  # fresh history table for this move's search

        core_search.set_use_nnue(self.use_dnn)

        root_fen = board.get_fen()
        best_move = None
        best_eval = -math.inf

        total_start = time.time()

        # Track best non-inf result from previous depths
        last_finite_move = None
        last_finite_eval = None
        last_finite_depth = 0

        # Track best move from previous depth for move ordering
        previous_best_move = None

        # iterate depths 1..self.depth
        for depth in range(1, self.depth + 1):
            core_search.reset_counters()
            core_search.clear_killers()
            core_search.clear_counters()
            # Keep TT between depths — iterative deepening benefits from
            # shallower results (TT probe already checks depth >= required)
            bar = tqdm(desc=f"Depth {depth}", total=None)

            # snapshot of nodes before this depth
            nodes_before = core_search.get_nodes_evaluated()

            root_board = chess.Board(root_fen)
            root_board.turn = chess.WHITE if color == 'white' else chess.BLACK
            all_moves = list(root_board.legal_moves)
            # Filter obvious blunders at root level
            moves = all_moves

            # Sort moves: put previous best move first for better alpha-beta cutoffs
            if previous_best_move and previous_best_move in moves:
                moves.remove(previous_best_move)
                moves = [previous_best_move] + moves

            # Compute root hash once (same board state for all root moves)
            root_ref = chess.Board(root_fen)
            root_ref.turn = chess.WHITE if color == 'white' else chess.BLACK
            pre_hash = core_search.compute_hash(root_ref)
            root_ck = root_ref.has_kingside_castling_rights(chess.WHITE)
            root_cq = root_ref.has_queenside_castling_rights(chess.WHITE)
            root_ck2 = root_ref.has_kingside_castling_rights(chess.BLACK)
            root_cq2 = root_ref.has_queenside_castling_rights(chess.BLACK)
            root_ep = root_ref.ep_square

            # ——— Aspiration Windows ———
            # For depth >= 3 with a finite previous eval, narrow the search window
            # around the previous depth's eval. On fail-high/fail-low, widen and retry.
            asp_delta = 0.5  # 50cp initial window radius
            if (depth >= 3
                and last_finite_eval is not None
                and not math.isinf(last_finite_eval)
                and abs(last_finite_eval) < 100000):
                asp_alpha = -math.inf # last_finite_eval - asp_delta
                asp_beta = math.inf # last_finite_eval + asp_delta
            else:
                # Shallow depth or no prev eval: full window
                asp_alpha = -math.inf
                asp_beta = math.inf

            asp_attempts = 0
            while True:
                # Sequential root search with proper alpha-beta bounds.
                # In negamax, we negate the child's value so the root always
                # maximizes. After evaluating the first (hopefully best) move,
                # alpha tightens and subsequent moves get pruned quickly.
                alpha = asp_alpha
                current_best = moves[0] if moves else None
                failed_high = False

                for uci in moves:
                    b = chess.Board(root_fen)
                    b.turn = chess.WHITE if color == 'white' else chess.BLACK
                    acc = Accumulator(); acc.init(b)
                    captured = b.piece_at(uci.to_square)
                    mover = b.piece_at(uci.from_square)

                    b.push(uci); acc.update(uci, captured)
                    root_key = core_search.update_hash_full(
                        pre_hash, uci, mover, captured,
                        root_ck, root_cq, root_ck2, root_cq2, root_ep, b)

                    # Negamax: negate child so we always maximize at root
                    # Pass -asp_beta as child's alpha (clamps to aspiration window)
                    val = -minimax(b, acc,
                                   depth - 1,
                                   -asp_beta, -alpha,
                                   color,
                                   root_key,
                                   depth)

                    # Update progress bar
                    nodes_after = core_search.get_nodes_evaluated()
                    delta = nodes_after - nodes_before
                    nodes_before = nodes_after
                    bar.update(delta)
                    bar.set_postfix({
                        'nodes': nodes_after,
                        'pruned': core_search.get_branches_pruned()
                    })

                    # Check if both are mate scores (>= 100000 means mate within horizon)
                    both_mates = abs(val) >= 100000 and abs(alpha) >= 100000

                    if val > alpha:
                        alpha = val
                        current_best = uci
                        if abs(val) >= 100000:
                            san_move = root_ref.san(uci)
                            if both_mates:
                                print(f"    Better mate found: {san_move} eval={val:.0f}")
                            else:
                                print(f"    First mate found: {san_move} eval={val:.0f}, continuing to find faster mate...")

                    # Beta cutoff at root: fail high
                    if alpha >= asp_beta:
                        failed_high = True
                        break

                # Aspiration result check
                asp_attempts += 1
                if not math.isinf(asp_alpha) and alpha <= asp_alpha:
                    # Fail low: widen alpha
                    asp_delta *= 4
                    if asp_attempts >= 2:
                        asp_alpha = -math.inf
                    else:
                        asp_alpha = last_finite_eval - asp_delta
                    print(f"    Aspiration fail-low at depth {depth}, widening to ({asp_alpha:.2f}, {asp_beta:.2f})")
                    continue
                elif failed_high or (not math.isinf(asp_beta) and alpha >= asp_beta):
                    # Fail high: widen beta
                    asp_delta *= 4
                    if asp_attempts >= 2:
                        asp_beta = math.inf
                    else:
                        asp_beta = last_finite_eval + asp_delta
                    print(f"    Aspiration fail-high at depth {depth}, widening to ({asp_alpha:.2f}, {asp_beta:.2f})")
                    continue
                else:
                    # Within window
                    break

            bar.close()
            best_move, best_eval = current_best, alpha

            # Track best move for next iteration's move ordering
            if best_move is not None:
                previous_best_move = best_move

            # Save this depth's result if eval is finite and not a mate score
            if not math.isinf(best_eval) and abs(best_eval) < 100000:
                last_finite_move = best_move
                last_finite_eval = best_eval
                last_finite_depth = depth

            # Handle inf values in printing (convert to White's perspective for consistency)
            display_eval = best_eval if color == 'white' else -best_eval
            eval_str = "inf" if math.isinf(display_eval) else f"{display_eval:.4f}"
            tt_h = core_search.get_tt_hits()
            tt_m = core_search.get_tt_misses()
            tt_total = tt_h + tt_m
            tt_pct = (100.0 * tt_h / tt_total) if tt_total > 0 else 0.0
            san_move = root_ref.san(best_move) if best_move else "None"
            print(f"Depth {depth} → best={san_move} eval={eval_str}  TT: {tt_h}/{tt_total} hits ({tt_pct:.1f}%)")

        # If final eval is inf (search bug), fall back to last reasonable result.
        # But keep mate scores (>=90000) — those are real forced mates.
        if math.isinf(best_eval) and last_finite_move is not None:
            print(f"Final eval is inf, using depth {last_finite_depth} result instead")
            best_move = last_finite_move
            best_eval = last_finite_eval

        elapsed = time.time() - total_start
        # Convert to absolute perspective: positive = White better, negative = Black better
        display_eval = best_eval if color == 'white' else -best_eval
        eval_str = "inf" if math.isinf(display_eval) else f"{display_eval:.4f}"
        print(f"AI search complete. Nodes: {core_search.get_nodes_evaluated()}, Pruned: {core_search.get_branches_pruned()}, Time: {elapsed:.2f}s")
        print(f"Eval: {eval_str}")

        if best_move is None:
            return None

        # map UCI back to your Move/Square classes
        mv = self._uci_to_move(board, best_move)
        piece = board.squares[mv.initial.row][mv.initial.col].piece
        return piece, mv

    def choose_promotion_piece(self, board, color, to_row, to_col):
        """
        Evaluate each promotion candidate (Q, R, B, N) with a 6-ply search.
        Returns the piece class with the highest evaluation.
        """
        from piece import Queen, Rook, Bishop, Knight

        piece_map = {
            chess.QUEEN: Queen,
            chess.ROOK: Rook,
            chess.BISHOP: Bishop,
            chess.KNIGHT: Knight,
        }

        root_fen = board.get_fen()
        best_cls = Queen  # default
        best_val = -math.inf

        for pt, cls in piece_map.items():
            # Build a board with this promotion applied
            test_board = chess.Board(root_fen)
            test_board.turn = chess.WHITE if color == 'white' else chess.BLACK

            # Place the promoted piece
            sq = (7 - to_row) * 8 + to_col  # convert row/col to python-chess square
            piece_obj = chess.Piece(pt, test_board.turn)
            test_board.set_piece_at(sq, piece_obj)

            # Flip turn to opponent for evaluation
            test_board.turn = not test_board.turn

            # Quick 6-ply search from opponent's perspective
            acc = Accumulator()
            acc.init(test_board)
            key = core_search.compute_hash(test_board)

            core_search.reset_counters()
            val = core_search.minimax(
                test_board, acc,
                6,                  # 6-ply depth
                -math.inf, math.inf,
                color,
                key,
                6                   # required_depth
            )
            # minimax returns from side-to-move perspective (opponent)
            # negate to get our perspective
            val = -val

            print(f"  Promotion eval: {cls.__name__} = {val:.2f}")

            if val > best_val:
                best_val = val
                best_cls = cls

        print(f"  AI chooses: {best_cls.__name__} (eval={best_val:.2f})")
        return best_cls

    def _evaluate_root(self, root_fen, uci, depth, maximize, ai_color):
        """Evaluate one root move via Cython minimax."""
        board = chess.Board(root_fen)
        board.turn = chess.WHITE if ai_color == 'white' else chess.BLACK
        acc = Accumulator(); acc.init(board)

        captured = board.piece_at(uci.to_square)
        mover = board.piece_at(uci.from_square)

        pre_hash = core_search.compute_hash(board)
        ck = board.has_kingside_castling_rights(chess.WHITE)
        cq = board.has_queenside_castling_rights(chess.WHITE)
        ck2 = board.has_kingside_castling_rights(chess.BLACK)
        cq2 = board.has_queenside_castling_rights(chess.BLACK)
        old_ep = board.ep_square

        board.push(uci); acc.update(uci, captured)
        root_key = core_search.update_hash_full(
            pre_hash, uci, mover, captured,
            ck, cq, ck2, cq2, old_ep, board)
        val = minimax(board, acc,
                depth - 1,
                -math.inf, math.inf,
                ai_color,
                root_key,
                depth)
        return val, uci

    def _fen_to_tensor(self, fen_str: str):
        """
        Convert FEN to HalfKP index tensors for model evaluation.
        """
        from nnue.halfkp import halfkp_indices_for_fen
        PAD_LEN = 30
        raw0, raw1 = halfkp_indices_for_fen(fen_str)
        idx0 = raw0 + [0] * (PAD_LEN - len(raw0))
        idx1 = raw1 + [0] * (PAD_LEN - len(raw1))
        t0 = torch.tensor(idx0[:PAD_LEN], dtype=torch.long).unsqueeze(0)
        t1 = torch.tensor(idx1[:PAD_LEN], dtype=torch.long).unsqueeze(0)
        return t0, t1