import pygame
from const import *
from board import Board
from dragger import Dragger
from square import Square
from config import Config
from promotion import PromotionModal

class Game:
    def __init__(self):
        self.next_player = 'white'
        self.hovered_sqr = None
        self.board = Board()
        self.dragger = Dragger()
        self.config = Config()
        self.game_over = False
        self.board_flipped = False  # False = white on bottom, True = black on bottom

        # Move history for undo/redo
        self.move_history = [(self.board.get_fen(), None)]  # (fen, move)
        self.history_index = 0

        # Promotion modal state
        self.promotion_pending = False
        self.promotion_pawn = None          # the Pawn piece object
        self.promotion_from_row = 0
        self.promotion_from_col = 0
        self.promotion_to_row = 0
        self.promotion_to_col = 0
        self.promotion_captured = None      # piece that was on the promotion square (if any)
        self.promotion_modal = PromotionModal()

    def screen_to_board(self, screen_row, screen_col):
        """Convert screen coordinates to actual board coordinates, accounting for flip."""
        if self.board_flipped:
            return 7 - screen_row, 7 - screen_col
        return screen_row, screen_col

    def board_to_screen(self, board_row, board_col):
        """Convert board coordinates to screen coordinates, accounting for flip."""
        if self.board_flipped:
            return 7 - board_row, 7 - board_col
        return board_row, board_col

    def add_move_to_history(self, move):
        """Add a move to history, truncating any moves after current position."""
        self.move_history = self.move_history[:self.history_index + 1]
        self.move_history.append((self.board.get_fen(), move))
        self.history_index = len(self.move_history) - 1

    def navigate_history(self, direction):
        """Navigate history: 'left' = back, 'right' = forward, 'up' = end, 'down' = start."""
        if direction == 'left':
            if self.history_index > 0:
                self.history_index -= 1
        elif direction == 'right':
            if self.history_index < len(self.move_history) - 1:
                self.history_index += 1
        elif direction == 'up':
            self.history_index = len(self.move_history) - 1
        elif direction == 'down':
            self.history_index = 0

        fen, move = self.move_history[self.history_index]
        import chess
        chess_board = chess.Board(fen)
        self.board.load_from_chess_board(chess_board)
        self.board.last_move = move if move is not None else None
        self.next_player = 'white' if chess_board.turn == chess.WHITE else 'black'
        self.dragger.undrag_piece()

    #Functions to show
    def show_bg(self, surface):
        theme = self.config.theme

        for row in range(ROWS):
            for col in range(COLS):
                screen_row, screen_col = self.board_to_screen(row, col)
                color = theme.bg.light if (row + col) % 2 == 0 else theme.bg.dark
                rect = (screen_col * SQSIZE, screen_row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(surface, color, rect)

                if screen_col == 0:
                    color = theme.bg.dark if screen_row % 2 == 0 else theme.bg.light
                    lbl = self.config.font.render(str(ROWS-row), 1, color)
                    lbl_pos = (5, 5 + screen_row * SQSIZE)
                    surface.blit(lbl, lbl_pos)

                if screen_row == 7:
                    color = theme.bg.dark if (row + col) % 2 == 0 else theme.bg.light
                    lbl = self.config.font.render(Square.get_alphacol(col), 1, color)
                    lbl_pos = (screen_col * SQSIZE + SQSIZE - 20, screen_row * SQSIZE + SQSIZE - 20)
                    surface.blit(lbl, lbl_pos)

    def show_pieces(self, surface):
        for row in range(ROWS):
            for col in range(COLS):
                if self.board.squares[row][col].has_piece():
                    piece = self.board.squares[row][col].piece

                    if piece is not self.dragger.piece:
                        piece.set_texture(size=80)
                        img = pygame.image.load(piece.texture)
                        screen_row, screen_col = self.board_to_screen(row, col)
                        img_center = (screen_col * SQSIZE + SQSIZE // 2, screen_row * SQSIZE + SQSIZE // 2)
                        piece.texture_rect = img.get_rect(center=img_center)
                        surface.blit(img, piece.texture_rect)
                        
    def show_moves(self, surface):
        theme = self.config.theme
        if self.dragger.dragging:
            piece = self.dragger.piece
            for move in piece.moves:
                color = theme.moves.light if (move.final.row + move.final.col) % 2 == 0 else theme.moves.dark
                screen_row, screen_col = self.board_to_screen(move.final.row, move.final.col)
                rect = (screen_col * SQSIZE, screen_row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(surface, color, rect)
                
    def show_last_move(self, surface):
        theme = self.config.theme
        if self.board.last_move:
            initial = self.board.last_move.initial
            final = self.board.last_move.final

            for pos in [initial, final]:
                color = theme.trace.light if (pos.row + pos.col) % 2 == 0 else theme.trace.dark
                screen_row, screen_col = self.board_to_screen(pos.row, pos.col)
                rect = (screen_col * SQSIZE, screen_row * SQSIZE, SQSIZE, SQSIZE)
                pygame.draw.rect(surface, color, rect)

    def show_hover(self, surface):
        if self.hovered_sqr:
            color = (180, 180, 180)
            screen_row, screen_col = self.board_to_screen(self.hovered_sqr.row, self.hovered_sqr.col)
            rect = (screen_col * SQSIZE, screen_row * SQSIZE, SQSIZE, SQSIZE)
            pygame.draw.rect(surface, color, rect, width=3)
        
    def next_turn(self):
        # print (f"next_turn + {self.next_player}")
        self.next_player = 'white' if self.next_player == 'black' else 'black'
        self.board.turn = self.next_player  # Keep board.turn in sync for FEN generation

        self.board.update_position_history(self.next_player)
        
        if self.board.is_checkmate(self.next_player):
            self.game_over = True
            winner = "Black" if self.next_player == "white" else "White"
            print(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate(self.next_player):
            self.game_over = True
            print("Stalemate! It's a draw.")
        elif self.board.half_move_clock >= 100:
            self.game_over = True
            print("Draw! 50-move rule.")
        elif self.board.is_threefold_repetition():
            self.game_over = True
            print("Draw! Threefold repetition.")
        elif self.board.is_insufficient_material():
            self.game_over = True
            print("Draw! Insufficient material.")
        elif self.board.is_in_check(self.next_player):
            self.config.check_sound.play()
            
    def show_check(self, surface):
        theme = self.config.theme
        for row in range(ROWS):
            for col in range(COLS):
                square = self.board.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    if piece.__class__.__name__ == "King" and self.board.is_in_check(piece.color):
                        color = theme.moves.light if (row + col) % 2 == 0 else theme.moves.dark
                        screen_row, screen_col = self.board_to_screen(row, col)
                        rect = (screen_col * SQSIZE, screen_row * SQSIZE, SQSIZE, SQSIZE)
                        pygame.draw.rect(surface, color, rect)
        
    def set_hover(self, screen_row, screen_col):
        board_row, board_col = self.screen_to_board(screen_row, screen_col)
        self.hovered_sqr = self.board.squares[board_row][board_col]
        
    def change_theme(self):
        self.config.change_theme()
        
    def play_sound(self, captured=False):
        if captured:
            self.config.capture_sound.play()
        else:
            self.config.move_sound.play()

    def show_promotion_modal(self, surface):
        """Draw the promotion modal if active."""
        self.promotion_modal.draw(surface)

    def reset(self):
        self.__init__()