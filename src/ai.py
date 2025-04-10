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

class ChessAI:
    def __init__(self, depth=3, use_dnn=False, model_path=None):
        """
        Initialize the chess AI.
        
        :param depth: How many plies to search.
        :param use_dnn: Whether to use a deep neural network for evaluation.
        :param model_path: Path to the pretrained model.
        """
        self.depth = depth
        self.use_dnn = use_dnn
        if self.use_dnn and model_path is not None and os.path.exists(model_path):
            self.model = torch.load(model_path)
            self.model.eval()
        else:
            self.model = None

        # Minimal opening book.
        # Keys are board positions in FEN notation (or another unique board representation)
        # and the values are lists of moves (in algebraic notation) recommended for that position.
        self.opening_book = {
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR": ["e2e4", "d2d4", "g1f3", "c2c4"],
            # Add more opening positions as needed.
        }

    def choose_move(self, board: Board, color: str):
        """
        Choose the best move for the given board and color.
        
        :param board: The Board object.
        :param color: The color to move ('white' or 'black').
        :return: A tuple (piece, move) representing the chosen move,
                 or None if no legal move exists.
        """
        # Reset stats for this move search.
        self.nodes_evaluated = 0
        self.branches_pruned = 0

        # Check if an opening book move is available.
        book_move = self.get_book_move(board, color)
        if book_move:
            print("Using book move.")
            return book_move

        start_time = time.time()
        legal_moves = self.get_all_legal_moves(board, color)
        if not legal_moves:
            return None  # No legal moves; game over or stalemate.
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

        # Print out search stats.
        print("AI move search complete.")
        print("Nodes evaluated:", self.nodes_evaluated)
        print("Branches pruned:", self.branches_pruned)
        print("Time taken (seconds):", end_time - start_time)
        print("Best evaluation for", color, ":", best_eval)

        return best_move

    def get_book_move(self, board: Board, color: str):
        """
        If available, select a move from the opening book based on the current board state.
        
        :param board: The Board object.
        :param color: The color to move.
        :return: A tuple (piece, move) from the book if found, otherwise None.
        """
        # Assume the Board class provides a get_fen() method.
        if hasattr(board, "get_fen"):
            fen = board.get_fen()
        else:
            # Fallback: use a simple string representation.
            fen = str(board.squares)
        if fen in self.opening_book:
            # Randomly choose one move from the opening book.
            chosen_notation = random.choice(self.opening_book[fen])
            # Find the corresponding legal move.
            legal_moves = self.get_all_legal_moves(board, color)
            for piece, move in legal_moves:
                # Assumes that Move objects support an algebraic() method.
                if hasattr(move, "algebraic") and move.algebraic() == chosen_notation:
                    return (piece, move)
        return None

    def minimax(self, board: Board, depth: int, alpha: float, beta: float, maximizing_player: bool, ai_color: str):
        """
        Minimax search with alpha-beta pruning.
        
        :param board: The current Board state.
        :param depth: Remaining depth to search.
        :param alpha: Alpha value for pruning.
        :param beta: Beta value for pruning.
        :param maximizing_player: Boolean flag indicating maximizer or minimizer turn.
        :param ai_color: The AI's color.
        :return: The evaluation score.
        """
        # Increment the nodes evaluated counter.
        self.nodes_evaluated += 1

        # Determine which color should move.
        current_color = ai_color if maximizing_player else ('white' if ai_color == 'black' else 'black')
        legal_moves = self.get_all_legal_moves(board, current_color)
        # Terminal condition: depth is zero or no moves exist.
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
        """
        Generate all legal moves for the given color.
        
        :param board: The Board state.
        :param color: The player's color.
        :return: A list of tuples (piece, move).
        """
        moves = []
        for row in range(8):  # Assuming board has 8 rows.
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
        """
        Order moves using a simple heuristic by simulating the move
        and evaluating the resulting board state.
        
        :param board: The current board.
        :param moves_list: List of legal moves as (piece, move) tuples.
        :param maximizing: Flag indicating whether ordering is for maximizing or minimizing.
        :param ai_color: The AI's color.
        :return: A sorted list of (piece, move) tuples.
        """
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
        Evaluate the board state from the perspective of ai_color.
        
        If DNN evaluation is enabled and a model is loaded, use it.
        Otherwise, use the enhanced evaluation function that includes
        material, piece-square tables, mobility, and king safety.
        
        :param board: The current board.
        :param ai_color: AI's color.
        :return: A numerical evaluation.
        """
        if self.use_dnn and self.model:
            board_tensor = self.board_to_tensor(board)
            with torch.no_grad():
                value = self.model(board_tensor)
            return value.item()
        else:
            return self.enhanced_evaluation(board, ai_color)

    def enhanced_evaluation(self, board: Board, ai_color: str):
        """
        An enhanced evaluation that includes material balance, positional bonuses
        using piece-square tables, mobility evaluation, and king safety.
        
        :param board: The current board state.
        :param ai_color: The perspective for evaluation ('white' or 'black').
        :return: A numerical evaluation of the board.
        """
        # Piece-square tables (from white's perspective).
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
            [-1,  0,  0.5, 1,   1, 0.5,  0, -1],
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

        # Evaluate material and positional factors for each piece on the board.
        for row in range(8):
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    # Base material score.
                    piece_score = piece.value

                    # Positional bonus from piece-square tables.
                    table_value = 0
                    piece_name = piece.__class__.__name__
                    if piece_name == "Pawn":
                        table_value = pawn_table[row][col] if piece.color == "white" else pawn_table[7 - row][col]
                    elif piece_name == "Knight":
                        table_value = knight_table[row][col] if piece.color == "white" else knight_table[7 - row][col]
                    elif piece_name == "Bishop":
                        table_value = bishop_table[row][col] if piece.color == "white" else bishop_table[7 - row][col]
                    elif piece_name == "Rook":
                        table_value = rook_table[row][col] if piece.color == "white" else rook_table[7 - row][col]
                    elif piece_name == "Queen":
                        table_value = queen_table[row][col] if piece.color == "white" else queen_table[7 - row][col]
                    elif piece_name == "King":
                        table_value = king_table[row][col] if piece.color == "white" else king_table[7 - row][col]

                    # Aggregate material and positional score.
                    if piece.color == "white":
                        score += piece_score + table_value
                    else:
                        score -= piece_score + table_value

        # Mobility evaluation: count legal moves for both sides.
        my_moves = len(self.get_all_legal_moves(board, ai_color))
        opponent = "white" if ai_color == "black" else "black"
        opp_moves = len(self.get_all_legal_moves(board, opponent))
        mobility_factor = 0.1
        score += mobility_factor * (my_moves - opp_moves)

        # King safety evaluation: check for friendly pawns adjacent to the king.
        def king_safety(board, color):
            safety_score = 0
            king_pos = None
            # Locate the king.
            for r in range(8):
                for c in range(8):
                    if board.squares[r][c].has_piece():
                        p = board.squares[r][c].piece
                        if p.__class__.__name__ == "King" and p.color == color:
                            king_pos = (r, c)
                            break
                if king_pos is not None:
                    break
            if king_pos is None:
                return safety_score  # Failsafe; should not normally happen.

            kr, kc = king_pos
            # Check adjacent squares for friendly pawns.
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = kr + dr, kc + dc
                    if 0 <= nr < 8 and 0 <= nc < 8:
                        if board.squares[nr][nc].has_piece():
                            neighbor = board.squares[nr][nc].piece
                            if neighbor.__class__.__name__ == "Pawn" and neighbor.color == color:
                                safety_score += 0.2
            return safety_score

        king_safety_bonus = king_safety(board, ai_color) - king_safety(board, opponent)
        score += king_safety_bonus

        # Return final evaluation from ai_color's perspective.
        return score if ai_color == "white" else -score

    def board_to_tensor(self, board: Board):
        """
        Convert the board into a tensor representation.
        
        Constructs a simple 8x8 tensor with each cell containing the piece's value.
        For kings, a value of 0 is used to avoid infinite values in the tensor.
        
        :param board: The current board.
        :return: A torch tensor of shape (1, 1, 8, 8).
        """
        board_array = []
        for row in range(8):
            row_vals = []
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    if isinstance(piece, King):
                        row_vals.append(0)
                    else:
                        row_vals.append(piece.value)
                else:
                    row_vals.append(0)
            board_array.append(row_vals)
        tensor = torch.tensor(board_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor

# Example usage (integration with your game loop is required):
# from board import Board
# board = Board()
# ai = ChessAI(depth=3, use_dnn=False)  # Set use_dnn=True with a valid model_path if available.
# chosen_move = ai.choose_move(board, 'black')
# if chosen_move:
#     piece, move = chosen_move
#     board.move(piece, move)  # Apply the move.
