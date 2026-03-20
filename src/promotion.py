import pygame
from const import SQSIZE
from piece import Queen, Rook, Bishop, Knight


class PromotionModal:
    """
    Chess.com-style vertical promotion dropdown.
    Appears centered on the promoting pawn's file.
    Expands downward for white (row 0), upward for black (row 7).
    """

    PIECE_CLASSES = [Queen, Rook, Bishop, Knight]

    def __init__(self):
        self.active = False
        self.color = None       # 'white' or 'black'
        self.col = 0            # file the pawn promoted on
        self.row = 0            # rank the pawn promoted to (0 or 7)
        self.board_flipped = False  # whether the board is flipped
        self.rects = []         # list of (pygame.Rect, piece_class) for click detection
        self.close_rect = None  # pygame.Rect for the X button

    def open(self, color, row, col, board_flipped=False):
        """Activate the modal for a promotion at (row, col)."""
        self.active = True
        self.color = color
        self.row = row
        self.col = col
        self.board_flipped = board_flipped

    def close(self):
        """Deactivate the modal."""
        self.active = False
        self.rects = []
        self.close_rect = None

    def draw(self, surface):
        """
        Draw the promotion dropdown on the surface.
        Returns nothing — click detection is handled by handle_click().
        """
        if not self.active:
            return

        # Draw semi-transparent overlay over the entire board
        overlay = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 80))
        surface.blit(overlay, (0, 0))

        # Transform board coordinates to screen coordinates when board is flipped
        screen_row = (7 - self.row) if self.board_flipped else self.row
        screen_col = (7 - self.col) if self.board_flipped else self.col

        # Determine direction: expand downward if promotion is at top of screen
        # When board is not flipped: white (row 0) is at top, expand down
        # When board is flipped: white (row 0) is at bottom, expand up
        expanding_down = (self.row == 0) if not self.board_flipped else (self.row == 7)

        # Modal dimensions
        modal_w = SQSIZE
        modal_h = SQSIZE * 4 + 30  # 4 piece slots + space for X button
        modal_x = screen_col * SQSIZE

        if expanding_down:
            modal_y = screen_row * SQSIZE
        else:
            modal_y = (screen_row + 1) * SQSIZE - modal_h

        # Draw modal background
        modal_rect = pygame.Rect(modal_x, modal_y, modal_w, modal_h)
        pygame.draw.rect(surface, (255, 255, 255), modal_rect, border_radius=4)
        pygame.draw.rect(surface, (180, 180, 180), modal_rect, width=1, border_radius=4)

        # Draw pieces
        self.rects = []
        pieces = self.PIECE_CLASSES

        for i, piece_cls in enumerate(pieces):
            piece = piece_cls(self.color)
            piece.set_texture(size=80)
            img = pygame.image.load(piece.texture)

            if expanding_down:
                slot_y = modal_y + i * SQSIZE
            else:
                slot_y = modal_y + modal_h - 30 - (4 - i) * SQSIZE

            slot_rect = pygame.Rect(modal_x, slot_y, SQSIZE, SQSIZE)
            img_center = slot_rect.center
            img_rect = img.get_rect(center=img_center)
            surface.blit(img, img_rect)
            self.rects.append((slot_rect, piece_cls))

        # Draw X button
        x_h = 26
        if expanding_down:
            x_y = modal_y + 4 * SQSIZE + 2
        else:
            x_y = modal_y + 2

        self.close_rect = pygame.Rect(modal_x, x_y, modal_w, x_h)
        font = pygame.font.SysFont('Arial', 18)
        x_label = font.render('X', True, (120, 120, 120))
        x_label_rect = x_label.get_rect(center=self.close_rect.center)
        surface.blit(x_label, x_label_rect)

    def handle_click(self, pos):
        """
        Check if a click at pos hits a piece, the X, or is outside.
        Returns:
            ('piece', piece_class)  - user clicked a piece
            ('cancel', None)        - user clicked X or outside
            (None, None)            - modal not active
        """
        if not self.active:
            return (None, None)

        x, y = pos

        # Check piece rects
        for rect, piece_cls in self.rects:
            if rect.collidepoint(x, y):
                return ('piece', piece_cls)

        # Check X button
        if self.close_rect and self.close_rect.collidepoint(x, y):
            return ('cancel', None)

        # Click outside modal - cancel
        return ('cancel', None)
