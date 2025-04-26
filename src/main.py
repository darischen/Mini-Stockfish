import pygame
import sys

from const import *
from game import Game
from move import Move
from square import Square
from piece import *
os.add_dll_directory(r"C:/Users/daris/AppData/Local/Programs/Python/Python310/Lib/site-packages/torch/lib")
from core_search import minimax
from core_search import set_use_nnue
import core_search
from ai import ChessAI

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.game = Game()
        self.ai = ChessAI(depth=5, use_dnn=False, model_path="./nnue/hidden128best1.0076e-2.pt")

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
            
            for event in pygame.event.get():
                
                #click event
                if event.type == pygame.MOUSEBUTTONDOWN:
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
                    if dragger.dragging:
                        dragger.update_mouse(event.pos)
                        
                        released_row = dragger.mouseY // SQSIZE
                        released_col = dragger.mouseX // SQSIZE
                        
                        dragger.piece.clear_moves()
                        board.calc_moves(dragger.piece, dragger.initial_row, dragger.initial_col, bool=True)

                        
                        initial = Square(dragger.initial_row, dragger.initial_col)
                        final = Square(released_row, released_col)
                        move = Move(initial, final)
                        
                        if board.valid_move(dragger.piece, move):
                            captured = board.squares[released_row][released_col].has_piece()
                            board.move(dragger.piece, move)
                            # print(f"{dragger.piece} moved to {released_row}, {released_col}")
                            
                            board.set_true_en_passant(dragger.piece)
                            
                            game.play_sound(captured)
                            
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
                    
                    # changing themes
                    if event.key == pygame.K_t:
                        game.change_theme()

                     # reset game
                    if event.key == pygame.K_r:
                        game.reset()
                        game = self.game
                        board = self.game.board
                        dragger = self.game.dragger
                    if event.key == pygame.K_a:
                        # Check if it's black's turn
                        if game.next_player == 'black':
                            ai_move = self.ai.choose_move(board, 'black')
                            if ai_move:
                                piece, move = ai_move
                                board.move(piece, move)
                                print(f"AI (black) moves: {move}")
                                # Optionally, play sound if needed, update en passant, etc.
                                board.set_true_en_passant(piece)
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