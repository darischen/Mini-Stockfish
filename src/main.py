import pygame
import sys
import chess

from const import *
from game import Game
from move import Move
from square import Square
from piece import *
from ai import ChessAI

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.game = Game()
        self.ai = ChessAI(depth=10,use_dnn=True)

    def mainloop(self):
        game = self.game
        screen = self.screen
        dragger = self.game.dragger
        board = self.game.board

        while True:
            game.show_bg(screen)
            game.show_last_move(screen)
            game.show_moves(screen)
            game.show_check(screen)
            game.show_pieces(screen)
            game.show_hover(screen)

            if dragger.dragging:
                dragger.update_blit(screen)

            # Draw promotion modal on top of everything
            if game.promotion_pending:
                game.show_promotion_modal(screen)

            for event in pygame.event.get():

                #click event
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game.promotion_pending:
                        continue  # board is frozen during promotion
                    dragger.update_mouse(event.pos)

                    screen_row = dragger.mouseY // SQSIZE
                    screen_col = dragger.mouseX // SQSIZE
                    clicked_row, clicked_col = game.screen_to_board(screen_row, screen_col)

                    if board.squares[clicked_row][clicked_col].has_piece():
                        piece = board.squares[clicked_row][clicked_col].piece

                        if piece.color == game.next_player:
                            piece.clear_moves()
                            board.calc_moves(piece, clicked_row, clicked_col, bool=True)
                            dragger.save_initial(event.pos)
                            dragger.drag_piece(piece)

                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)

                elif event.type == pygame.MOUSEMOTION:
                    if game.promotion_pending:
                        continue  # board is frozen during promotion

                    motion_row = event.pos[1] // SQSIZE
                    motion_col = event.pos[0] // SQSIZE

                    game.set_hover(motion_row, motion_col)

                    if dragger.dragging:
                        dragger.piece.clear_moves()
                        board_row, board_col = game.screen_to_board(dragger.initial_row, dragger.initial_col)
                        board.calc_moves(dragger.piece, board_row, board_col, bool=True)
                        dragger.update_mouse(event.pos)
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_moves(screen)
                        game.show_check(screen)
                        game.show_pieces(screen)
                        game.show_hover(screen)
                        dragger.update_blit(screen)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if game.promotion_pending:
                        # Handle promotion modal click
                        action, piece_cls = game.promotion_modal.handle_click(event.pos)
                        if action == 'piece':
                            # Complete the promotion
                            board.complete_promotion(
                                game.promotion_to_row,
                                game.promotion_to_col,
                                piece_cls
                            )
                            game.promotion_modal.close()
                            game.promotion_pending = False

                            board.set_true_en_passant(game.promotion_pawn)
                            game.play_sound(game.promotion_captured is not None)

                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_pieces(screen)

                            game.next_turn()
                            promo_move = Move(Square(game.promotion_from_row, game.promotion_from_col),
                                            Square(game.promotion_to_row, game.promotion_to_col))
                            game.add_move_to_history(promo_move)
                        elif action == 'cancel':
                            # Revert the move
                            board.revert_promotion(
                                game.promotion_pawn,
                                game.promotion_from_row,
                                game.promotion_from_col,
                                game.promotion_to_row,
                                game.promotion_to_col,
                                game.promotion_captured
                            )
                            game.promotion_modal.close()
                            game.promotion_pending = False

                            game.show_bg(screen)
                            game.show_pieces(screen)

                    elif dragger.dragging:
                        dragger.update_mouse(event.pos)

                        screen_row = dragger.mouseY // SQSIZE
                        screen_col = dragger.mouseX // SQSIZE
                        released_row, released_col = game.screen_to_board(screen_row, screen_col)

                        dragger.piece.clear_moves()
                        board_row, board_col = game.screen_to_board(dragger.initial_row, dragger.initial_col)
                        board.calc_moves(dragger.piece, board_row, board_col, bool=True)

                        initial = Square(board_row, board_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        if board.valid_move(dragger.piece, move):
                            captured = board.squares[released_row][released_col].piece
                            uci_move = chess.Move(chess.parse_square(f"{chr(97+initial.col)}{8-initial.row}"),
                                                 chess.parse_square(f"{chr(97+final.col)}{8-final.row}"))
                            board.move(dragger.piece, move)
                            board.chess_board.push(uci_move)  # Keep chess board in sync

                            # Check if this is a promotion
                            if board.check_promotion(dragger.piece, final):
                                # Enter promotion-pending state
                                game.promotion_pending = True
                                game.promotion_pawn = dragger.piece
                                game.promotion_from_row = dragger.initial_row
                                game.promotion_from_col = dragger.initial_col
                                game.promotion_to_row = released_row
                                game.promotion_to_col = released_col
                                game.promotion_captured = captured
                                game.promotion_modal.open(
                                    dragger.piece.color,
                                    released_row,
                                    released_col,
                                    game.board_flipped
                                )
                                dragger.undrag_piece()
                                # Don't switch turns yet — wait for selection
                            else:
                                board.set_true_en_passant(dragger.piece)
                                game.play_sound(captured is not None)

                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_pieces(screen)

                                game.next_turn()
                                game.add_move_to_history(move)
                        else:
                            # if the released square is the same as the initial square don't play the sound
                            if (dragger.initial_row, dragger.initial_col) != (released_row, released_col):
                                game.config.illegal_sound.play()
                            dragger.piece.clear_moves()
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)

                    dragger.undrag_piece()


                elif event.type == pygame.KEYDOWN:
                    if game.promotion_pending:
                        continue  # block keyboard during promotion

                    # changing themes
                    if event.key == pygame.K_t:
                        game.change_theme()

                    # flip board perspective
                    if event.key == pygame.K_f:
                        game.board_flipped = not game.board_flipped

                    # load FEN
                    if event.key == pygame.K_e:
                        fen = input("Enter FEN: ")
                        try:
                            chess_board = chess.Board(fen)
                            board.load_from_chess_board(chess_board)
                            game.next_player = 'white' if chess_board.turn == chess.WHITE else 'black'
                            self.ai.reset()
                            print("Board loaded from FEN")
                        except Exception as ex:
                            print(f"Invalid FEN: {ex}")

                     # reset game
                    if event.key == pygame.K_r:
                        game.reset()
                        self.ai.reset()  # Reset AI state (out_of_book flag)
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger
                    if event.key == pygame.K_a:
                        # Check if it's black's turn
                        if game.next_player == 'black':
                            ai_move = self.ai.choose_move(board, 'black')
                            if ai_move:
                                _, mv = ai_move
                                piece = board.squares[mv.initial.row][mv.initial.col].piece
                                captured = board.squares[mv.final.row][mv.final.col].has_piece()

                                # Compute SAN from synchronized chess board
                                uci_move = chess.Move(chess.parse_square(f"{chr(97+mv.initial.col)}{8-mv.initial.row}"),
                                                     chess.parse_square(f"{chr(97+mv.final.col)}{8-mv.final.row}"))
                                san_move = board.chess_board.san(uci_move)

                                board.move(piece, mv)
                                board.chess_board.push(uci_move)  # Keep chess board in sync

                                # Check if AI move is a promotion
                                if board.check_promotion(piece, mv.final):
                                    promo_cls = self.ai.choose_promotion_piece(
                                        board, 'black',
                                        mv.final.row, mv.final.col
                                    )
                                    board.complete_promotion(
                                        mv.final.row, mv.final.col, promo_cls
                                    )

                                print(f"AI (black) moves: {mv} ({san_move})")
                                board.set_true_en_passant(piece)
                                game.play_sound(captured)
                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_pieces(screen)
                                game.show_check(screen)
                                game.show_hover(screen)
                                game.next_turn()
                                game.add_move_to_history(mv)
                            else:
                                print("AI found no legal moves for black.")
                        else:
                            print("It's not black's turn. AI move skipped.")
                    if event.key == pygame.K_s:
                        # Check if it's white's turn
                        if game.next_player == 'white':
                            ai_move = self.ai.choose_move(board, 'white')
                            if ai_move:
                                _, mv = ai_move
                                piece = board.squares[mv.initial.row][mv.initial.col].piece
                                captured = board.squares[mv.final.row][mv.final.col].has_piece()

                                # Compute SAN from synchronized chess board
                                uci_move = chess.Move(chess.parse_square(f"{chr(97+mv.initial.col)}{8-mv.initial.row}"),
                                                     chess.parse_square(f"{chr(97+mv.final.col)}{8-mv.final.row}"))
                                san_move = board.chess_board.san(uci_move)

                                board.move(piece, mv)
                                board.chess_board.push(uci_move)  # Keep chess board in sync

                                # Check if AI move is a promotion
                                if board.check_promotion(piece, mv.final):
                                    promo_cls = self.ai.choose_promotion_piece(
                                        board, 'white',
                                        mv.final.row, mv.final.col
                                    )
                                    board.complete_promotion(
                                        mv.final.row, mv.final.col, promo_cls
                                    )

                                print(f"AI (white) moves: {mv} ({san_move})")
                                board.set_true_en_passant(piece)
                                game.play_sound(captured)
                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_pieces(screen)
                                game.show_check(screen)
                                game.show_hover(screen)
                                game.next_turn()
                                game.add_move_to_history(mv)
                            else:
                                print("AI found no legal moves for white.")
                        else:
                            print("It's not white's turn. AI move skipped.")

                    # Navigate move history
                    if event.key == pygame.K_LEFT:
                        game.navigate_history('left')
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_pieces(screen)
                        game.show_check(screen)
                    elif event.key == pygame.K_RIGHT:
                        game.navigate_history('right')
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_pieces(screen)
                        game.show_check(screen)
                    elif event.key == pygame.K_UP:
                        game.navigate_history('up')
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_pieces(screen)
                        game.show_check(screen)
                    elif event.key == pygame.K_DOWN:
                        game.navigate_history('down')
                        game.show_bg(screen)
                        game.show_last_move(screen)
                        game.show_pieces(screen)
                        game.show_check(screen)

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()

main = Main()
main.mainloop()
