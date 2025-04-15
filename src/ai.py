# ai.py
import math
import os
import time
import random
import torch  # Ensure PyTorch is installed if you plan to use a DNN
import torch.nn as nn
from board import Board  # Board implementation
from move import Move   # Move class (ensure it includes the algebraic() method)
from piece import King  # King and other piece classes
from square import Square  # Square operations
import copy
import concurrent.futures
import bitboard  # Our bitboard module (should provide square_bb, popcount, king_attacks, etc.)
import numpy as np
from nnue.nnue_train import NNUEModel

# We now import our enhanced feature function;
# For example, you can place the enhanced_fen_to_features in a helper module,
# but here we define it inline.

def enhanced_fen_to_features(fen):
    """
    Convert a FEN string into an enriched feature vector.
    Starts with a 768-dimensional one-hot encoding (64 squares x 12 piece types)
    and then appends three extra features:
        - White king safety: count of friendly pawns adjacent to the white king.
        - Black king safety: same for the black king.
        - Mobility: total number of legal moves.
    Final dimension: 768 + 3 = 771.
    """
    board = None
    try:
        # Use python-chess to parse the fen.
        import chess
        board = chess.Board(fen)
    except Exception as e:
        raise ValueError("Invalid FEN string") from e

    base_features = np.zeros(64 * 12, dtype=np.float32)
    piece_to_index = {
         chess.PAWN: 0,
         chess.KNIGHT: 1,
         chess.BISHOP: 2,
         chess.ROOK: 3,
         chess.QUEEN: 4,
         chess.KING: 5,
    }
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            idx = piece_to_index[piece.piece_type]
            if not piece.color:  # chess.BLACK is False, chess.WHITE is True
                idx += 6
            base_features[square * 12 + idx] = 1.0

    # Extra Feature 1: King Safety for each color.
    def king_safety(color):
        king_sq = board.king(color)
        if king_sq is None:
            return 0.0
        safety = 0.0
        # Define adjacent offsets (simulate king's move offsets)
        offsets = [-9, -8, -7, -1, 1, 7, 8, 9]
        for offset in offsets:
            neighbor = king_sq + offset
            if neighbor < 0 or neighbor >= 64:
                continue
            if abs((neighbor % 8) - (king_sq % 8)) > 1:
                continue
            n_piece = board.piece_at(neighbor)
            if n_piece is not None and n_piece.color == color and n_piece.piece_type == chess.PAWN:
                safety += 1.0
        return safety

    white_king_safety = king_safety(chess.WHITE)
    black_king_safety = king_safety(chess.BLACK)

    # Extra Feature 2: Mobility (number of legal moves).
    mobility = float(len(list(board.legal_moves)))
    
    extra_features = np.array([white_king_safety, black_king_safety, mobility], dtype=np.float32)
    full_features = np.concatenate((base_features, extra_features))
    return full_features

# Use our enhanced features as the main feature extraction.
def fen_to_features(fen):
    return enhanced_fen_to_features(fen)

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
        if self.use_dnn and model_path is not None and os.path.exists(model_path):
            self.model = NNUEModel(input_size=771)
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.model = None

        # Minimal opening book.
        self.opening_book = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR": ["e2e4", "d2d4", "g1f3", "c2c4"],
        }

    def choose_move(self, board: Board, color: str):
        """
        Choose the best move for the given board and color.
        :param board: The Board object.
        :param color: The color to move ('white' or 'black').
        :return: A tuple (piece, move) representing the chosen move,
                 or None if no legal move exists.
        """
        self.nodes_evaluated = 0
        self.branches_pruned = 0

        book_move = self.get_book_move(board, color)
        if book_move:
            print("Using book move.")
            return book_move

        start_time = time.time()
        legal_moves = self.get_all_legal_moves(board, color)
        if not legal_moves:
            return None
        best_move = None
        best_eval = -math.inf if color == 'white' else math.inf

        for piece, move in legal_moves:
            move_history = board.make_move(piece, move)
            eval_value = self.minimax(board, self.depth - 1, -math.inf, math.inf, False, color)
            board.undo_move(move_history)
            if color == 'white' and eval_value > best_eval:
                best_eval = eval_value
                best_move = (piece, move)
            elif color == 'black' and eval_value < best_eval:
                best_eval = eval_value
                best_move = (piece, move)

        end_time = time.time()
        print("AI move search complete.")
        print("Nodes evaluated:", self.nodes_evaluated)
        print("Branches pruned:", self.branches_pruned)
        print("Time taken (seconds):", end_time - start_time)
        print("Best evaluation for", color, ":", best_eval)
        return best_move

    def get_book_move(self, board: Board, color: str):
        if hasattr(board, "get_fen"):
            fen = board.get_fen()
        else:
            fen = str(board.squares)
        if fen in self.opening_book:
            chosen_notation = random.choice(self.opening_book[fen])
            legal_moves = self.get_all_legal_moves(board, color)
            for piece, move in legal_moves:
                if hasattr(move, "algebraic") and move.algebraic() == chosen_notation:
                    return (piece, move)
        return None

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool, ai_color: str):
        self.nodes_evaluated += 1

        current_color = ai_color if maximizing_player else ('white' if ai_color == 'black' else 'black')
        legal_moves = self.get_all_legal_moves(board, current_color)
        if depth == 0 or not legal_moves:
            return self.evaluate(board, ai_color)

        if maximizing_player:
            max_eval = -math.inf
            ordered_moves = self.order_moves(board, legal_moves, maximizing=True, ai_color=ai_color)
            for piece, move in ordered_moves:
                move_history = board.make_move(piece, move)
                eval_value = self.minimax(board, depth - 1, alpha, beta, False, ai_color)
                board.undo_move(move_history)
                max_eval = max(max_eval, eval_value)
                alpha = max(alpha, eval_value)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return max_eval
        else:
            min_eval = math.inf
            ordered_moves = self.order_moves(board, legal_moves, maximizing=False, ai_color=ai_color)
            for piece, move in ordered_moves:
                move_history = board.make_move(piece, move)
                eval_value = self.minimax(board, depth - 1, alpha, beta, True, ai_color)
                board.undo_move(move_history)
                min_eval = min(min_eval, eval_value)
                beta = min(beta, eval_value)
                if beta <= alpha:
                    self.branches_pruned += 1
                    break
            return min_eval

    def get_all_legal_moves(self, board: Board, color: str):
        moves = []
        for row in range(8):
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    if piece.color == color:
                        piece.clear_moves()
                        board.calc_moves(piece, row, col, bool=True)
                        for move in piece.moves:
                            if board.valid_move(piece, move):
                                moves.append((piece, move))
        return moves

    def order_moves(self, board: Board, moves_list, maximizing: bool, ai_color: str):
        scored_moves = []
        for piece, move in moves_list:
            move_history = board.make_move(piece, move)
            score = self.evaluate(board, ai_color)
            board.undo_move(move_history)
            scored_moves.append(((piece, move), score))
        scored_moves.sort(key=lambda x: x[1], reverse=maximizing)
        ordered = [move for move, score in scored_moves]
        return ordered

    def evaluate(self, board: Board, ai_color: str):
        """
        Evaluate the board state. If NNUE is enabled and a model is loaded, use it.
        Otherwise, use our bitboard-based evaluation.
        """
        if self.use_dnn and self.model:
            board_tensor = self.board_to_tensor(board)
            with torch.no_grad():
                value = self.model(board_tensor)
            return value.item()
        else:
            return self.evaluate_bitboard(board, ai_color)

    def evaluate_bitboard(self, board: Board, ai_color: str):
        """
        Evaluate using a bitboard-based method.
        (This method remains unchanged.)
        """
        bitboards = {}
        for row in range(8):
            for col in range(8):
                sq_index = row * 8 + col
                square = board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    key = (piece.color, piece.__class__.__name__)
                    bitboards[key] = bitboards.get(key, 0) | (1 << sq_index)

        piece_values = {
            "Pawn": 100,
            "Knight": 320,
            "Bishop": 330,
            "Rook": 500,
            "Queen": 900,
            "King": 20000
        }
        pawn_table = [
            [0,    0,    0,    0,    0,    0,    0,    0],
            [0.5,  1,    1,   -2,   -2,    1,    1,   0.5],
            [0.5, -0.5, -1,    0,    0,   -1,  -0.5,  0.5],
            [0,    0,    0,    2,    2,    0,    0,    0],
            [0.5,  0.5,   1,   2.5,  2.5,   1,   0.5,  0.5],
            [1,    1,     2,    3,    3,    2,    1,    1],
            [5,    5,     5,   -5,   -5,    5,    5,    5],
            [0,    0,     0,    0,    0,    0,    0,    0]
        ]
        knight_table = [
            [-5, -4, -3, -3, -3, -3, -4, -5],
            [-4, -2,  0,  0,  0,  0, -2, -4],
            [-3,  0,  1,  1.5, 1.5, 1,  0, -3],
            [-3,  0.5, 1.5, 2,  2,  1.5, 0.5, -3],
            [-3,  0,  1.5, 2,  2,  1.5, 0, -3],
            [-3,  0.5, 1,  1.5, 1.5, 1,  0.5, -3],
            [-4, -2,  0,  0.5, 0.5, 0, -2, -4],
            [-5, -4, -3, -3, -3, -3, -4, -5]
        ]
        bishop_table = [
            [-2, -1, -1, -1, -1, -1, -1, -2],
            [-1,  0,  0,  0,  0,  0,  0, -1],
            [-1,  0,  0.5, 1,   1,  0.5,  0, -1],
            [-1, 0.5, 0.5, 1,   1, 0.5, 0.5, -1],
            [-1,  0,   1,   1,   1,   1,  0, -1],
            [-1,  1,   1,   1,   1,   1,  1, -1],
            [-1, 0.5,   0,   0,   0,   0, 0.5, -1],
            [-2, -1,  -1,  -1,  -1,  -1, -1, -2]
        ]
        rook_table = [
            [0,   0,   0,   0,   0,   0,   0,   0],
            [0.5, 1,   1,   1,   1,   1,   1,  0.5],
            [-0.5, 0,   0,   0,   0,   0,   0, -0.5],
            [-0.5, 0,   0,   0,   0,   0,   0, -0.5],
            [-0.5, 0,   0,   0,   0,   0,   0, -0.5],
            [-0.5, 0,   0,   0,   0,   0,   0, -0.5],
            [-0.5, 0,   0,   0,   0,   0,   0, -0.5],
            [0,   0,   0,   0,   0,   0,   0,   0]
        ]
        queen_table = [
            [-2, -1, -1, -0.5, -0.5, -1, -1, -2],
            [-1,  0,  0,    0,    0,  0,  0, -1],
            [-1,  0,  0.5, 0.5,  0.5, 0.5, 0, -1],
            [-0.5,0,  0.5, 0.5,  0.5, 0.5, 0, -0.5],
            [0,   0,  0.5, 0.5,  0.5, 0.5, 0, -0.5],
            [-1, 0.5, 0.5, 0.5,  0.5, 0.5, 0, -1],
            [-1,  0,  0.5,   0,    0,  0,  0, -1],
            [-2, -1, -1, -0.5, -0.5, -1, -1, -2]
        ]
        king_table = [
            [-3, -4, -4, -5, -5, -4, -4, -3],
            [-3, -4, -4, -5, -5, -4, -4, -3],
            [-3, -4, -4, -5, -5, -4, -4, -3],
            [-3, -4, -4, -5, -5, -4, -4, -3],
            [-2, -3, -3, -4, -4, -3, -3, -2],
            [-1, -2, -2, -2, -2, -2, -2, -1],
            [2,   2,   0,   0,   0,   0,   2,   2],
            [2,   3,   1,   0,   0,   1,   3,   2]
        ]
        score = 0
        piece_values = {
            "Pawn": 100,
            "Knight": 320,
            "Bishop": 330,
            "Rook": 500,
            "Queen": 900,
            "King": 20000
        }
        piece_tables = {
            "Pawn": pawn_table,
            "Knight": knight_table,
            "Bishop": bishop_table,
            "Rook": rook_table,
            "Queen": queen_table,
            "King": king_table
        }
        for (color_key, ptype), bb in bitboards.items():
            if ptype not in piece_values:
                continue
            for sq in self._bits_iter(bb):
                row = sq // 8
                col = sq % 8
                bonus = piece_tables[ptype][row][col] if color_key == "white" else piece_tables[ptype][7 - row][col]
                val = piece_values[ptype] + bonus
                if color_key == ai_color:
                    score += val
                else:
                    score -= val

        opponent = "white" if ai_color == "black" else "black"
        mobility_factor = 0.1
        score += mobility_factor * (len(self.get_all_legal_moves(board, ai_color)) - len(self.get_all_legal_moves(board, opponent)))
        king_sq = None
        for row in range(8):
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece() and square.piece.__class__.__name__ == "King" and square.piece.color == ai_color:
                    king_sq = row * 8 + col
                    break
            if king_sq is not None:
                break
        if king_sq is not None:
            king_adjacent = bitboard.king_attacks(king_sq)
            friendly_pawns = bitboards.get((ai_color, "Pawn"), 0)
            king_safety_bonus = 0.2 * bitboard.popcount(king_adjacent & friendly_pawns)
            score += king_safety_bonus

        return score if ai_color == "white" else -score

    def _bits_iter(self, bb: int):
        while bb:
            lsb = bb & -bb
            yield lsb.bit_length() - 1
            bb &= bb - 1

    def board_to_tensor(self, board: Board):
        """
        Convert the board into a tensor representation using the enhanced feature extraction.
        This function assumes that either the board provides a get_fen() method or that board.squares can be processed.
        """
        if hasattr(board, "get_fen"):
            fen = board.get_fen()
        else:
            # Fallback: reconstruct FEN from board.squares if necessary.
            raise NotImplementedError("Board does not provide get_fen method.")
        features = fen_to_features(fen)  # This now returns a 771-dimensional vector.
        tensor = torch.from_numpy(features).unsqueeze(0)  # Shape: [1, 771]
        return tensor

# Example usage (integration with your game loop is required):
# from board import Board
# board = Board()
# ai = ChessAI(depth=3, use_dnn=True, model_path="nnue_model.pth")  # Ensure you have a valid model path
# chosen_move = ai.choose_move(board, 'black')
# if chosen_move:
#     piece, move = chosen_move
#     board.move(piece, move)  # Apply the move.