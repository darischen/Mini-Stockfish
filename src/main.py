import pygame
import sys

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
        self.ai = ChessAI(depth=10, use_dnn=True)

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

                    clicked_row = dragger.mouseY // SQSIZE
                    clicked_col = dragger.mouseX // SQSIZE

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
                        board.calc_moves(dragger.piece, dragger.initial_row, dragger.initial_col, bool=True)
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

                        released_row = dragger.mouseY // SQSIZE
                        released_col = dragger.mouseX // SQSIZE

                        dragger.piece.clear_moves()
                        board.calc_moves(dragger.piece, dragger.initial_row, dragger.initial_col, bool=True)

                        initial = Square(dragger.initial_row, dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)

                        if board.valid_move(dragger.piece, move):
                            captured = board.squares[released_row][released_col].piece
                            board.move(dragger.piece, move)

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
                                    released_col
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
                                board.move(piece, mv)

                                # Check if AI move is a promotion
                                if board.check_promotion(piece, mv.final):
                                    promo_cls = self.ai.choose_promotion_piece(
                                        board, 'black',
                                        mv.final.row, mv.final.col
                                    )
                                    board.complete_promotion(
                                        mv.final.row, mv.final.col, promo_cls
                                    )

                                print(f"AI (black) moves: {mv}")
                                board.set_true_en_passant(piece)
                                game.play_sound(captured)
                                game.show_bg(screen)
                                game.show_last_move(screen)
                                game.show_pieces(screen)
                                game.show_check(screen)
                                game.show_hover(screen)
                                game.next_turn()
                            else:
                                print("AI found no legal moves for black.")
                        else:
                            print("It's not black's turn. AI move skipped.")

                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            pygame.display.update()

main = Main()
main.mainloop()
