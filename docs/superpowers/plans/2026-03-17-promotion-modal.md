# Promotion Modal Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a chess.com-style promotion modal that lets human players choose which piece to promote to, and uses a 6-ply search for AI promotion decisions.

**Architecture:** When a pawn reaches the promotion rank, the board enters a frozen "promotion pending" state. A vertical dropdown of 4 piece options (using existing PNG textures) appears centered on the pawn's file, expanding downward for white (row 0) or upward for black (row 7). Clicking a piece completes the promotion; clicking X or outside cancels and reverts the pawn. The AI uses a 6-ply minimax search to evaluate each promotion candidate and picks the best.

**Tech Stack:** Python, Pygame, Cython (core_search module)

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/promotion.py` | **Create** | `PromotionModal` class — renders the dropdown, handles click detection, loads textures |
| `src/board.py` | **Modify** | Change `check_promotion()` to return bool instead of auto-promoting; add `complete_promotion()` and `revert_promotion()` methods |
| `src/game.py` | **Modify** | Add promotion-pending state fields; add `show_promotion_modal()` rendering method |
| `src/main.py` | **Modify** | Wire promotion modal into event loop — freeze board, handle modal clicks, handle AI promotion |
| `src/ai.py` | **Modify** | Add `choose_promotion_piece()` method using 6-ply search |

---

## Chunk 1: Core Promotion Infrastructure

### Task 1: Create PromotionModal class

**Files:**
- Create: `src/promotion.py`

- [ ] **Step 1: Create `src/promotion.py` with the `PromotionModal` class**

This class manages the visual dropdown and click detection. It does NOT own game state — it just renders and reports clicks.

```python
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
        self.rects = []         # list of (pygame.Rect, piece_class) for click detection
        self.close_rect = None  # pygame.Rect for the X button

    def open(self, color, row, col):
        """Activate the modal for a promotion at (row, col)."""
        self.active = True
        self.color = color
        self.row = row
        self.col = col

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

        # Determine direction: white promotes at row 0 → expand downward
        #                       black promotes at row 7 → expand upward
        expanding_down = (self.row == 0)

        # Modal dimensions
        modal_w = SQSIZE
        modal_h = SQSIZE * 4 + 30  # 4 piece slots + space for X button
        modal_x = self.col * SQSIZE

        if expanding_down:
            modal_y = self.row * SQSIZE
        else:
            modal_y = (self.row + 1) * SQSIZE - modal_h

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
            ('piece', piece_class)  — user clicked a piece
            ('cancel', None)        — user clicked X or outside
            (None, None)            — modal not active
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

        # Click outside modal — cancel
        return ('cancel', None)
```

- [ ] **Step 2: Commit**

```bash
git add src/promotion.py
git commit -m "feat: add PromotionModal class for chess.com-style piece selection"
```

---

### Task 2: Modify board.py — promotion detection and reversion

**Files:**
- Modify: `src/board.py:284-287` (`check_promotion` method)
- Modify: `src/board.py:256` (call site in `move()`)

- [ ] **Step 1: Change `check_promotion` to return a bool instead of auto-promoting**

In `src/board.py`, replace the existing `check_promotion` method (lines 284-287):

```python
# BEFORE:
def check_promotion(self, piece, final):
    if final.row == 0 or final.row == 7:
        self.config.promotion_sound.play()
        self.squares[final.row][final.col].piece = Queen(piece.color)

# AFTER:
def check_promotion(self, piece, final):
    """Return True if this pawn move is a promotion (reached rank 0 or 7)."""
    return isinstance(piece, Pawn) and (final.row == 0 or final.row == 7)
```

- [ ] **Step 2: Update the call site in `move()` to not call check_promotion automatically**

In `src/board.py`, the `move()` method around line 254-256 currently does:

```python
        if isinstance(piece, Pawn):
            ...
            else:
                self.check_promotion(piece, final)
```

Change to:

```python
        if isinstance(piece, Pawn):
            ...
            # Promotion is now handled externally by the game loop
            # (check_promotion just returns True/False, doesn't auto-promote)
```

Remove the `self.check_promotion(piece, final)` call from inside `move()`. Promotion will be detected and handled by the caller.

- [ ] **Step 3: Add `complete_promotion()` and `revert_promotion()` methods**

Add these methods to the `Board` class in `src/board.py`:

```python
def complete_promotion(self, row, col, piece_cls):
    """Replace the pawn at (row, col) with the chosen piece."""
    color = self.squares[row][col].piece.color
    self.squares[row][col].piece = piece_cls(color)
    self.config.promotion_sound.play()

def revert_promotion(self, piece, from_row, from_col, to_row, to_col, captured_piece):
    """
    Undo a pawn move to the promotion rank.
    Move the pawn back from (to_row, to_col) to (from_row, from_col),
    and restore any captured piece at (to_row, to_col).
    """
    self.squares[from_row][from_col].piece = piece
    self.squares[to_row][to_col].piece = captured_piece
    self.last_move = None
    piece.moved = False
```

- [ ] **Step 4: Commit**

```bash
git add src/board.py
git commit -m "feat: refactor check_promotion to support modal selection and add revert"
```

---

### Task 3: Add promotion state to game.py

**Files:**
- Modify: `src/game.py`

- [ ] **Step 1: Add promotion-pending state fields to `Game.__init__`**

In `src/game.py`, add these fields to `__init__` after `self.game_over = False`:

```python
        # Promotion modal state
        self.promotion_pending = False
        self.promotion_pawn = None          # the Pawn piece object
        self.promotion_from_row = 0
        self.promotion_from_col = 0
        self.promotion_to_row = 0
        self.promotion_to_col = 0
        self.promotion_captured = None      # piece that was on the promotion square (if any)
```

- [ ] **Step 2: Add import and create PromotionModal instance**

At the top of `src/game.py`, add the import:

```python
from promotion import PromotionModal
```

In `__init__`, add:

```python
        self.promotion_modal = PromotionModal()
```

- [ ] **Step 3: Add `show_promotion_modal()` method**

Add this method to `Game`:

```python
def show_promotion_modal(self, surface):
    """Draw the promotion modal if active."""
    self.promotion_modal.draw(surface)
```

- [ ] **Step 4: Commit**

```bash
git add src/game.py
git commit -m "feat: add promotion-pending state and modal to Game class"
```

---

## Chunk 2: Main Loop Integration and AI Promotion

### Task 4: Wire promotion modal into the main event loop

**Files:**
- Modify: `src/main.py`

This is the most important task — it wires everything together. The key changes:

1. After a valid pawn move to the promotion rank, enter promotion-pending state instead of switching turns
2. When promotion is pending, render the modal and handle clicks
3. On piece selection, complete the promotion and switch turns
4. On cancel, revert the move

- [ ] **Step 1: Add promotion import**

At the top of `src/main.py`, the existing imports already include `from piece import *` which covers Queen, Rook, Bishop, Knight.

No new import needed.

- [ ] **Step 2: Modify the MOUSEBUTTONUP handler to detect promotions**

In `src/main.py`, replace the current MOUSEBUTTONUP block (lines 78-116) with:

```python
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
                            if (dragger.initial_row, dragger.initial_col) != (released_row, released_col):
                                game.config.illegal_sound.play()
                            dragger.piece.clear_moves()
                            game.show_bg(screen)
                            game.show_last_move(screen)
                            game.show_moves(screen)
                            game.show_pieces(screen)

                    dragger.undrag_piece()
```

- [ ] **Step 3: Freeze the board during promotion — block MOUSEBUTTONDOWN and MOUSEMOTION**

In the MOUSEBUTTONDOWN handler (line 39 area), add a guard at the top:

```python
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if game.promotion_pending:
                        continue  # board is frozen during promotion
                    dragger.update_mouse(event.pos)
                    ...
```

Similarly for MOUSEMOTION:

```python
                elif event.type == pygame.MOUSEMOTION:
                    if game.promotion_pending:
                        continue  # board is frozen during promotion
                    ...
```

- [ ] **Step 4: Add promotion modal rendering to the main draw loop**

In the main render section (lines 25-34), add the modal rendering after pieces:

```python
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
```

- [ ] **Step 5: Commit**

```bash
git add src/main.py
git commit -m "feat: wire promotion modal into main event loop with freeze and cancel"
```

---

### Task 5: Add AI promotion piece selection

**Files:**
- Modify: `src/ai.py`

- [ ] **Step 1: Add `choose_promotion_piece()` method to ChessAI**

Add this method to the `ChessAI` class in `src/ai.py`:

```python
def choose_promotion_piece(self, board, color, to_row, to_col):
    """
    Evaluate each promotion candidate (Q, R, B, N) with a 6-ply search.
    Returns the piece class with the highest evaluation.
    """
    import chess
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
```

- [ ] **Step 2: Modify the AI move handler in main.py to handle promotions**

In `src/main.py`, in the `K_a` key handler (lines 132-154), after `board.move(piece, mv)`:

```python
                    if event.key == pygame.K_a:
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
```

- [ ] **Step 3: Commit**

```bash
git add src/ai.py src/main.py
git commit -m "feat: add AI promotion piece selection using 6-ply search"
```

---

### Task 6: Handle edge case — captured piece tracking fix

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Fix captured piece tracking in the promotion flow**

In the current code, `captured = board.squares[released_row][released_col].has_piece()` returns a bool. But `revert_promotion()` needs the actual piece object to restore it. The plan already addresses this in Task 4 Step 2 by changing it to:

```python
captured = board.squares[released_row][released_col].piece  # piece object or None
```

Verify this is consistent: `game.play_sound(captured is not None)` uses `is not None` instead of truthy check (since piece objects are always truthy).

- [ ] **Step 2: Verify the AI handler also uses the correct captured tracking**

In the AI handler, `captured = board.squares[mv.final.row][mv.final.col].has_piece()` is fine as-is since AI promotions don't need reversion. No change needed.

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: promotion modal complete — human and AI support"
```
