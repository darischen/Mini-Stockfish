from const import *
from square import Square
from piece import *
from move import Move
from sound import Sound
from config import Config
import copy
import os

class Board:
    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0] for col in range(COLS)]
        self.last_move = None
        self._create()
        self._add_pieces('white')
        self._add_pieces('black')
        self.config = Config()
        self.position_history = {}
        self.update_position_history('white')
        self.half_move_clock = 0
        
    def board_signature(self, next_player):
        signature = [next_player]
        for row in range(ROWS):
            for col in range(COLS):
                square = self.squares[row][col]
                if square.has_piece():
                    piece = square.piece
                    # Use the first letter of the piece's name; for Knight, use 'N'
                    letter = piece.__class__.__name__[0]
                    if piece.__class__.__name__ == "Knight":
                        letter = "N"
                    # Represent white pieces as uppercase, black as lowercase.
                    letter = letter.upper() if piece.color == 'white' else letter.lower()
                    signature.append(letter)
                else:
                    signature.append('.')
        return ''.join(signature)

    def update_position_history(self, next_player):
        sig = self.board_signature(next_player)
        if sig in self.position_history:
            self.position_history[sig] += 1
        else:
            self.position_history[sig] = 1

    def is_threefold_repetition(self):
        for count in self.position_history.values():
            if count >= 3:
                return True
        return False
    
    def is_insufficient_material(self):
        white_extra = []
        black_extra = []
        # Iterate over all squares and collect non-king pieces for each side.
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if isinstance(piece, King):
                        continue
                    # Append the piece to the appropriate list.
                    if piece.color == 'white':
                        white_extra.append(piece)
                    else:
                        black_extra.append(piece)
                        
        # If any pawn is present, then there is sufficient material.
        for p in white_extra + black_extra:
            # Assuming Pawn is defined in piece.py.
            if isinstance(p, Pawn):
                return False

        # Define allowed extra material for a draw: either no extra piece
        # or a single knight or a single bishop.
        def allowed(piece_list):
            if len(piece_list) == 0:
                return True
            if len(piece_list) == 1 and (isinstance(piece_list[0], Knight) or isinstance(piece_list[0], Bishop)):
                return True
            return False

        return allowed(white_extra) and allowed(black_extra)

        
    @staticmethod
    def moves_to_str(moves):
        return ", ".join(f"(({m.initial.row},{m.initial.col}) -> ({m.final.row},{m.final.col}))" for m in moves)
    
    def is_stalemate(self, color):
        # If the king is in check, it's not a stalemate.
        if self.is_in_check(color):
            return False

        # For every piece of the given color, calculate legal moves.
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if piece.color == color:
                        piece.clear_moves()
                        self.calc_moves(piece, row, col, bool=True)
                        if piece.moves:
                            # If at least one legal move is found, not stalemate.
                            return False

        # No legal moves and king is not in check: stalemate!
        return True
    
    def make_move(self, piece, move):
        move_history = {
            'piece': piece,
            'initial': (move.initial.row, move.initial.col),
            'final': (move.final.row, move.final.col),
            'captured': None,
            'piece_moved': piece.moved,
            'en_passant': False,
            'captured_location': None,
        }
        if isinstance(piece, Pawn) and (move.initial.col != move.final.col) and self.squares[move.final.row][move.final.col].isempty():
            move_history['en_passant'] = True
            captured_row = move.initial.row
            captured_col = move.final.col
            move_history['captured'] = self.squares[captured_row][captured_col].piece
            move_history['captured_location'] = (captured_row, captured_col)
            self.squares[captured_row][captured_col].piece = None
        else:
            move_history['captured'] = self.squares[move.final.row][move.final.col].piece

        self.squares[move.final.row][move.final.col].piece = piece
        self.squares[move.initial.row][move.initial.col].piece = None
        piece.moved = True

        return move_history

    def undo_move(self, move_history):
        piece = move_history['piece']
        init_row, init_col = move_history['initial']
        final_row, final_col = move_history['final']
        self.squares[init_row][init_col].piece = piece
        self.squares[final_row][final_col].piece = None
        if move_history.get('en_passant'):
            captured_row, captured_col = move_history['captured_location']
            self.squares[captured_row][captured_col].piece = move_history['captured']
        else:
            self.squares[final_row][final_col].piece = move_history['captured']

        piece.moved = move_history['piece_moved']
    
    def is_in_check(self, color):
    # Find the king's position for the given color.
        king_position = None
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if isinstance(piece, King) and piece.color == color:
                        king_position = (row, col)
                        break
            if king_position:
                break
        if king_position is None:
            # In a valid game, the king should always be present.
            return False
        king_row, king_col = king_position
        # Check if the king's square is attacked by any enemy piece.
        return self.is_square_attacked(king_row, king_col, color)

    def is_checkmate(self, color):
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if piece.color == color:
                        piece.clear_moves()
                        self.calc_moves(piece, row, col, bool=True)
                        # Debug (optional):
                        # print(f"{piece} at ({row},{col}) moves: {self.moves_to_str(piece.moves)}")

        # 2) If the king of 'color' is not currently in check, it cannot be checkmate
        if not self.is_in_check(color):
            return False

        # 3) If the king is in check, see if any piece of 'color' has at least one legal move
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    piece = self.squares[row][col].piece
                    if piece.color == color:
                        if piece.moves:
                            # Found at least one legal move => not checkmate
                            return False

        # 4) No legal moves remain, and the king is in check => checkmate
        self.config.checkmate_sound.play()
        return True

    def is_move_safe(self, piece, move):
        # Simulate the move
        move_history = self.make_move(piece, move)
        # Check if the king of the moving color is in check after the move
        safe = not self.is_in_check(piece.color)
        # Undo the move to restore the original board state
        self.undo_move(move_history)
        print(f"Move {move} is {'safe' if safe else 'unsafe'}")
        print(f"is_move_safe returns {safe} for move {move}")
        return safe

    def is_square_attacked(self, row, col, color):
        for r in range(ROWS):
            for c in range(COLS):
                if self.squares[r][c].has_piece():
                    piece = self.squares[r][c].piece
                    if piece.color != color:
                        # Handle enemy kings separately to avoid recursion.
                        if isinstance(piece, King):
                            if abs(r - row) <= 1 and abs(c - col) <= 1:
                                return True
                        else:
                            piece.clear_moves()
                            self.calc_moves(piece, r, c, bool=False)
                            for move in piece.moves:
                                if move.final.row == row and move.final.col == col:
                                    return True
        return False
    
    def move(self, piece, move, testing=False):
        initial = move.initial
        final = move.final
        
        was_capture = False
        # Check if destination square originally has a piece (normal capture)
        if not self.squares[final.row][final.col].isempty():
            was_capture = True
        # For pawn moves, if the move is diagonal then it is a capture (even en passant)
        if isinstance(piece, Pawn) and (final.col != initial.col):
            was_capture = True

        en_passant_empty = self.squares[final.row][final.col].isempty()

        self.squares[initial.row][initial.col].piece = None
        self.squares[final.row][final.col].piece = piece

        if isinstance(piece, Pawn):
            # If the pawn moves two squares forward, set en_passant to True
            diff = final.col - initial.col
            if diff != 0 and en_passant_empty:
                self.squares[initial.row][initial.col + diff].piece = None
                self.squares[final.row][final.col].piece = piece
                if not testing:
                    sound = Sound(
                        os.path.join('../assets/sounds/capture.mp3'))
                    sound.play()
            
            else:
                self.check_promotion(piece, final)

        if isinstance(piece, King):
            if self.castling(initial, final) and not testing:
                diff = final.col - initial.col
                rook = piece.left_rook if (diff < 0) else piece.right_rook
                self.move(rook, rook.moves[-1])
                self.config.castle_sound.play()
        
        if not testing:
            if was_capture or isinstance(piece, Pawn):
                self.half_move_clock = 0
            else:
                self.half_move_clock += 1
                
            piece.moved = True
            piece.clear_moves()
            self.last_move = move

    def check_promotion(self, piece, final):
        if final.row == 0 or final.row == 7:
            self.config.promotion_sound.play()
            self.squares[final.row][final.col].piece = Queen(piece.color)

    def castling(self, initial, final):
        return abs(initial.col - final.col) == 2

    def set_true_en_passant(self, piece):
        
        if not isinstance(piece, Pawn):
            return

        for row in range(ROWS):
            for col in range(COLS):
                if isinstance(self.squares[row][col].piece, Pawn):
                    self.squares[row][col].piece.en_passant = False
        
        piece.en_passant = True

    def in_check(self, piece, move):
        # Apply the move and record state
        move_history = self.make_move(piece, move)

        # Locate the king for piece's color
        king_position = None
        for row in range(ROWS):
            for col in range(COLS):
                if self.squares[row][col].has_piece():
                    p = self.squares[row][col].piece
                    if isinstance(p, King) and p.color == piece.color:
                        king_position = (row, col)
                        break
            if king_position:
                break

        check_found = False
        if king_position:
            king_row, king_col = king_position
            for row in range(ROWS):
                for col in range(COLS):
                    if self.squares[row][col].has_piece():
                        enemy = self.squares[row][col].piece
                        if enemy.color != piece.color:
                            enemy.clear_moves()
                            # Use bool=False to avoid recursive check validations.
                            self.calc_moves(enemy, row, col, bool=False)
                            for m in enemy.moves:
                                if m.final.row == king_row and m.final.col == king_col:
                                    check_found = True
                                    break
                            if check_found:
                                break
                if check_found:
                    break

        # Undo the simulated move regardless of the outcome
        self.undo_move(move_history)
        return check_found

    def calc_moves(self, piece, row, col, bool=True):
        
        def pawn_moves():
            steps = 1 if piece.moved else 2
            start = row + piece.dir
            end = row + (piece.dir * (1 + steps))
            for possible_move_row in range(start, end, piece.dir):
                if Square.in_range(possible_move_row):
                    if self.squares[possible_move_row][col].isempty():
                        
                        initial = Square(row, col)
                        final = Square(possible_move_row, col)
                        
                        move = Move(initial, final)

                        if bool:
                            if self.is_move_safe(piece, move):
                                piece.add_move(move)
                                valid_moves = [str(m) for m in piece.moves]
                                print("Valid moves for this piece:")
                                for valid_move in valid_moves:
                                    print(valid_move)
                        else:
                            piece.add_move(move)
                    else:
                        break
                else:
                    break

            
            possible_move_row = row + piece.dir
            possible_move_cols = [col-1, col+1]
            for possible_move_col in possible_move_cols:
                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].has_enemy_piece(piece.color):
                        
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        
                        move = Move(initial, final)
                        
                        if bool:
                            if self.is_move_safe(piece, move):
                                piece.add_move(move)
                        else:
                            piece.add_move(move)

            
            r = 3 if piece.color == 'white' else 4
            fr = 2 if piece.color == 'white' else 5
            
            if Square.in_range(col-1) and row == r:
                if self.squares[row][col-1].has_enemy_piece(piece.color):
                    p = self.squares[row][col-1].piece
                    if isinstance(p, Pawn):
                        if p.en_passant:
                            
                            initial = Square(row, col)
                            final = Square(fr, col-1, p)
                            
                            move = Move(initial, final)
                            
                            if bool:
                                if self.is_move_safe(piece, move):
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)
            
            if Square.in_range(col+1) and row == r:
                if self.squares[row][col+1].has_enemy_piece(piece.color):
                    p = self.squares[row][col+1].piece
                    if isinstance(p, Pawn):
                        if p.en_passant:
                            
                            initial = Square(row, col)
                            final = Square(fr, col+1, p)
                            
                            move = Move(initial, final)
                            
                            if bool:
                                if self.is_move_safe(piece, move):   
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)
        
        def knight_moves():
            possible_moves = [
                (row-2, col+1),
                (row-1, col+2),
                (row+1, col+2),
                (row+2, col+1),
                (row+2, col-1),
                (row+1, col-2),
                (row-1, col-2),
                (row-2, col-1),
            ]

            for possible_move in possible_moves:
                possible_move_row, possible_move_col = possible_move

                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].isempty_or_enemy(piece.color):
                        
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        
                        move = Move(initial, final)
                        
                        if bool:
                            if self.is_move_safe(piece, move):
                                piece.add_move(move)
                        else:
                            piece.add_move(move)
                            
        def king_moves():
            adjs = [
                (row-1, col+0),
                (row-1, col+1),
                (row+0, col+1),
                (row+1, col+1),
                (row+1, col+0),
                (row+1, col-1),
                (row+0, col-1),
                (row-1, col-1),
            ]
            
            for possible_move in adjs:
                possible_move_row, possible_move_col = possible_move
                if Square.in_range(possible_move_row, possible_move_col):
                    if self.squares[possible_move_row][possible_move_col].isempty_or_enemy(piece.color):
                        initial = Square(row, col)
                        final = Square(possible_move_row, possible_move_col)
                        move = Move(initial, final)
                        if self.is_move_safe(piece, move):
                            piece.add_move(move)

            
            if not piece.moved:
                left_rook = self.squares[row][0].piece
                if isinstance(left_rook, Rook):
                    if not left_rook.moved:
                        for c in range(1, 4):
                            
                            if self.squares[row][c].has_piece():
                                break

                            if c == 3:
                                piece.left_rook = left_rook
                                
                                initial = Square(row, 0)
                                final = Square(row, 3)
                                moveR = Move(initial, final)
                                
                                initial = Square(row, col)
                                final = Square(row, 2)
                                moveK = Move(initial, final)

                                if bool:
                                    if self.is_move_safe(piece, moveK) and self.is_move_safe(left_rook, moveR):
                                        left_rook.add_move(moveR)   
                                        piece.add_move(moveK)
                                else:
                                    left_rook.add_move(moveR)
                                    piece.add_move(moveK)
                
                right_rook = self.squares[row][7].piece
                if isinstance(right_rook, Rook):
                    if not right_rook.moved:
                        for c in range(5, 7):
                
                            if self.squares[row][c].has_piece():
                                break

                            if c == 6:
                                piece.right_rook = right_rook
 
                                initial = Square(row, 7)
                                final = Square(row, 5)
                                moveR = Move(initial, final)

                                initial = Square(row, col)
                                final = Square(row, 6)
                                moveK = Move(initial, final)

                
                                if bool:
                                    if self.is_move_safe(piece, moveK) and self.is_move_safe(right_rook, moveR):
                                        right_rook.add_move(moveR)
                                        piece.add_move(moveK)
                                else:
                                    right_rook.add_move(moveR)
                                    piece.add_move(moveK)
        
        def straightline_moves(incrs):
            for incr in incrs:
                row_incr, col_incr = incr
                possible_move_row = row + row_incr
                possible_move_col = col + col_incr

                while True:
                    if Square.in_range(possible_move_row, possible_move_col):
                        
                        initial = Square(row, col)
                        final_piece = self.squares[possible_move_row][possible_move_col].piece
                        final = Square(possible_move_row, possible_move_col, final_piece)
                        
                        move = Move(initial, final)

                        if self.squares[possible_move_row][possible_move_col].isempty():
                            if bool:
                                if self.is_move_safe(piece, move):
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)

                        elif self.squares[possible_move_row][possible_move_col].has_enemy_piece(piece.color):
                            if bool:
                                if self.is_move_safe(piece, move):
                                    piece.add_move(move)
                            else:
                                piece.add_move(move)
                            break

                        elif self.squares[possible_move_row][possible_move_col].has_team_piece(piece.color):
                            break
                    
                    else:
                        break
                    
                    possible_move_row = possible_move_row + row_incr
                    possible_move_col = possible_move_col + col_incr
        
        if isinstance(piece, Pawn):
            pawn_moves()
            
        elif isinstance(piece, Knight):
            knight_moves()
            
        elif isinstance(piece, Bishop):
            straightline_moves([
                (-1, 1),
                (-1, -1),
                (1, 1),
                (1, -1),
            ])
            
        elif isinstance(piece, Rook):
            straightline_moves([
                (-1, 0),
                (0, 1),
                (1, 0),
                (0, -1),
            ])
            
        elif isinstance(piece, Queen):
            straightline_moves([
                (-1, 1),
                (-1, -1),
                (1, 1),
                (1, -1),
                (-1, 0),
                (0, 1),
                (1, 0),
                (0, -1)
            ])
            
        elif isinstance(piece, King):
            king_moves()
            
    
    def _create(self):
        for row in range(ROWS):
            for col in range(COLS):
                self.squares[row][col] = Square(row, col)
    
    def valid_move(self, piece, move):
        return move in piece.moves
    
    def _add_pieces(self, color):
        row_pawn, row_other = (6, 7) if color == 'white' else (1, 0)
        
        # Create all pawns
        for col in range(COLS):
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))
            
        # Create all Knights
        self.squares[row_other][1] = Square(row_other, 1, Knight(color))
        self.squares[row_other][6] = Square(row_other, 6, Knight(color))
        
        # Create all Bishops
        self.squares[row_other][2] = Square(row_other, 2, Bishop(color))
        self.squares[row_other][5] = Square(row_other, 5, Bishop(color))
        
        # Create all Rooks
        self.squares[row_other][0] = Square(row_other, 0, Rook(color))
        self.squares[row_other][7] = Square(row_other, 7, Rook(color))
        
        # Create Queen and King
        self.squares[row_other][3] = Square(row_other, 3, Queen(color))
        self.squares[row_other][4] = Square(row_other, 4, King(color))