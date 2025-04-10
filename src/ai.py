# ai.py
import math
import os
import time
import torch  # Ensure PyTorch is installed if you plan to use a DNN
import torch.nn as nn
from board import Board  # Board implementation
from move import Move   # Move class
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
        # Increment the nodes evaluated counter each time minimax is called.
        self.nodes_evaluated += 1

        # Determine which color should move on this level.
        current_color = ai_color if maximizing_player else ('white' if ai_color == 'black' else 'black')
        legal_moves = self.get_all_legal_moves(board, current_color)
        # Terminal condition: depth limit reached or no moves available.
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
        # For maximizing, higher scores first; for minimizing, lower scores first.
        scored_moves.sort(key=lambda x: x[1], reverse=maximizing)
        ordered = [move for move, score in scored_moves]
        return ordered

    def evaluate(self, board: Board, ai_color: str):
        """
        Evaluate the board state from the perspective of ai_color.
        
        If DNN evaluation is enabled and a model is loaded, use it.
        Otherwise, use an advanced evaluation function that considers material
        and positional factors.
        
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
            return self.advanced_evaluation(board, ai_color)

    def advanced_evaluation(self, board: Board, ai_color: str):
        """
        An advanced evaluation that considers both the material value and positional
        factors such as central control.
        
        :param board: The current board.
        :param ai_color: The AI's color.
        :return: The board's evaluation score.
        """
        material_score = 0
        positional_score = 0
        # Define the center region (rows 2-5 and cols 2-5)
        center_rows = range(2, 6)
        center_cols = range(2, 6)
        
        for row in range(8):
            for col in range(8):
                square = board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    # Skip king evaluation.
                    if isinstance(piece, King):
                        continue
                    material_score += piece.value
                    # Add a bonus for pieces in the center.
                    if row in center_rows and col in center_cols:
                        bonus = 0.5 if piece.color == ai_color else -0.5
                        positional_score += bonus
        score = material_score + positional_score
        # Adjust for board values being from white's perspective.
        return score if ai_color == 'white' else -score

    def board_to_tensor(self, board: Board):
        """
        Convert the board into a tensor representation.
        
        Builds a simple 8x8 tensor with each cell containing the piece value.
        More advanced representations could include separate channels per piece type.
        
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
                    # For kings, set to 0 to avoid math.inf in the tensor.
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
# ai = ChessAI(depth=3, use_dnn=False)  # Set use_dnn=True and provide a model_path if available.
# chosen_move = ai.choose_move(board, 'black')
# if chosen_move:
#     piece, move = chosen_move
#     board.move(piece, move)  # Apply the move.
