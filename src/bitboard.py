"""
bitboard.py

A simplified bitboard system inspired by Stockfish’s C++ implementation.
It represents the board as a 64-bit integer and precomputes utility tables
such as SquareDistance. It includes functions to generate sliding moves for
rooks and bishops (without magic bitboard optimizations) and to compute
pseudo-attacks for knights and kings. This module serves as a starting point;
for a full-speed engine, consider a more optimized (and low-level) solution.
"""

import math

# -- Constants and Utility Functions --

NUM_SQUARES = 64  # 8 x 8 chessboard

def square_bb(square: int) -> int:
    """
    Returns a bitboard with only the bit corresponding to the given square set.
    Square is an integer in 0..63 where:
      a1 = 0, b1 = 1, ..., h1 = 7,
      a2 = 8, …, h8 = 63.
    """
    return 1 << square

def popcount(bb: int) -> int:
    """Count the number of 1s (occupied squares) in the bitboard."""
    return bb.bit_count()

def pretty(bb: int) -> str:
    """
    Returns an ASCII representation of a bitboard.
    The top rank (8) is printed first and the bottom rank (1) is printed last.
    """
    s = "+---+---+---+---+---+---+---+---+\n"
    for rank in reversed(range(8)):  # rank 7 down to 0
        for file in range(8):
            square = rank * 8 + file
            s += "| X " if bb & square_bb(square) else "|   "
        s += "| " + str(rank + 1) + "\n+---+---+---+---+---+---+---+---+\n"
    s += "  a   b   c   d   e   f   g   h\n"
    return s

def file_of(square: int) -> int:
    return square % 8

def rank_of(square: int) -> int:
    return square // 8

def on_board(square: int) -> bool:
    return 0 <= square < NUM_SQUARES

# -- Precomputed Table: Square Distance --

# SquareDistance[s1][s2] gives the maximum of file and rank distances between squares s1 and s2.
SquareDistance = [[0] * NUM_SQUARES for _ in range(NUM_SQUARES)]
for s1 in range(NUM_SQUARES):
    r1, f1 = rank_of(s1), file_of(s1)
    for s2 in range(NUM_SQUARES):
        r2, f2 = rank_of(s2), file_of(s2)
        SquareDistance[s1][s2] = max(abs(r1 - r2), abs(f1 - f2))

# -- Sliding Attack Generators (Simplified) --

# Define directional offsets.
# Note: In our board representation, adding 8 moves one rank up;
# adding 1 moves one file to the right. We must be careful with file wrap-around.
DIRECTIONS_ROOK = [8, -8, 1, -1]          # North, South, East, West
DIRECTIONS_BISHOP = [9, 7, -7, -9]         # NorthEast, NorthWest, SouthEast, SouthWest

def rook_attacks(square: int, occupied: int) -> int:
    """
    Generate the bitboard of rook moves from square, given an occupancy bitboard.
    This method iterates in each rook direction until hitting the board edge or an occupied square.
    """
    attacks = 0
    for offset in DIRECTIONS_ROOK:
        s = square
        while True:
            next_sq = s + offset
            # Check board boundaries. For horizontal moves, ensure file does not wrap.
            if not on_board(next_sq):
                break
            # For East moves, if current file is 7, cannot move right.
            if offset == 1 and file_of(s) == 7:
                break
            # For West moves, if current file is 0, cannot move left.
            if offset == -1 and file_of(s) == 0:
                break
            attacks |= square_bb(next_sq)
            if occupied & square_bb(next_sq):
                break
            s = next_sq
    return attacks

def bishop_attacks(square: int, occupied: int) -> int:
    """
    Generate the bitboard of bishop moves from square, given an occupancy bitboard.
    Iterates in each diagonal direction.
    """
    attacks = 0
    for offset in DIRECTIONS_BISHOP:
        s = square
        while True:
            next_sq = s + offset
            # Check board boundaries.
            if not on_board(next_sq):
                break
            # Check file wrap-around: the absolute change in file must be 1.
            if abs(file_of(next_sq) - file_of(s)) != 1:
                break
            attacks |= square_bb(next_sq)
            if occupied & square_bb(next_sq):
                break
            s = next_sq
    return attacks

def queen_attacks(square: int, occupied: int) -> int:
    """
    Queen attacks are the union of rook and bishop attacks.
    """
    return rook_attacks(square, occupied) | bishop_attacks(square, occupied)

# -- Precomputed Tables for Knight and King Moves --

KNIGHT_OFFSETS = [17, 15, 10, 6, -6, -10, -15, -17]
KING_OFFSETS = [8, 1, -1, -8, 9, 7, -7, -9]

def knight_attacks(square: int) -> int:
    """
    Generate the bitboard for all knight moves from the given square.
    Checks for board boundaries and avoids wrapping around files.
    """
    attacks = 0
    r, f = rank_of(square), file_of(square)
    for offset in KNIGHT_OFFSETS:
        next_sq = square + offset
        if not on_board(next_sq):
            continue
        nr, nf = rank_of(next_sq), file_of(next_sq)
        # A valid knight move is L-shaped: (2,1) or (1,2)
        if (abs(nr - r), abs(nf - f)) in [(2, 1), (1, 2)]:
            attacks |= square_bb(next_sq)
    return attacks

def king_attacks(square: int) -> int:
    """
    Generate the bitboard for king moves (one step in any direction) from square.
    """
    attacks = 0
    r, f = rank_of(square), file_of(square)
    for offset in KING_OFFSETS:
        next_sq = square + offset
        if not on_board(next_sq):
            continue
        nr, nf = rank_of(next_sq), file_of(next_sq)
        if abs(nr - r) <= 1 and abs(nf - f) <= 1:
            attacks |= square_bb(next_sq)
    return attacks

# -- Example Usage and Testing --

if __name__ == "__main__":
    # Example: Represent a piece on e4.
    # Convention: a1 = 0, b1 = 1, ..., h1 = 7, a2 = 8, ... h8 = 63.
    # For example, square e4: file 'e' (index 4; since a=0, b=1, ..., h=7) and rank 4 (rank index 3)
    e4 = 3 * 8 + 4  # 3*8=24, plus 4 equals 28 (so e4 is square 28)
    
    print("Bitboard with a piece on e4:")
    bb_e4 = square_bb(e4)
    print(pretty(bb_e4))
    
    # Suppose an empty board (occupied = 0)
    occupied = 0
    
    print("Rook attacks from e4 (empty board):")
    print(pretty(rook_attacks(e4, occupied)))
    
    print("Bishop attacks from e4 (empty board):")
    print(pretty(bishop_attacks(e4, occupied)))
    
    print("Queen attacks from e4 (empty board):")
    print(pretty(queen_attacks(e4, occupied)))
    
    print("Knight attacks from e4:")
    print(pretty(knight_attacks(e4)))
    
    print("King attacks from e4:")
    print(pretty(king_attacks(e4)))
    
    # Demonstrate the precomputed square distance between two squares.
    # For example, distance between a1 (square 0) and h8 (square 63)
    print("Square distance between a1 and h8:", SquareDistance[0][63])
