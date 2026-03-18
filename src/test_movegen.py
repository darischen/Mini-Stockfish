"""Perft test: verify C++ movegen matches python-chess legal moves."""
import ctypes
import os
import sys
import chess

# Load the DLL
dll_path = os.path.join(os.path.dirname(__file__), "nnue", "build", "Release", "chess_movegen.dll")
lib = ctypes.CDLL(dll_path)

# Define CBoard struct
class CBoard(ctypes.Structure):
    _fields_ = [
        ("pieces", (ctypes.c_uint64 * 7) * 2),
        ("occupied", ctypes.c_uint64),
        ("occupied_co", ctypes.c_uint64 * 2),
        ("side_to_move", ctypes.c_int),
        ("ep_square", ctypes.c_int),
        ("castling", ctypes.c_uint8),
    ]

# Setup function signatures
lib.cboard_from_bitboards.argtypes = [ctypes.POINTER(CBoard), ctypes.POINTER(ctypes.c_uint64),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_uint8]
lib.cboard_from_bitboards.restype = None

lib.cboard_legal_moves.argtypes = [ctypes.POINTER(CBoard), ctypes.POINTER(ctypes.c_uint32)]
lib.cboard_legal_moves.restype = ctypes.c_int

lib.movegen_init.argtypes = []
lib.movegen_init.restype = None

lib.movegen_init()

def sync_board(board):
    """Convert python-chess board to CBoard."""
    cb = CBoard()
    bb14 = (ctypes.c_uint64 * 14)()

    # White pieces (occupied_co[chess.WHITE] = occupied_co[True] = occupied_co[1])
    white_occ = board.occupied_co[chess.WHITE]
    black_occ = board.occupied_co[chess.BLACK]

    bb14[0]  = board.pawns   & white_occ
    bb14[1]  = board.knights & white_occ
    bb14[2]  = board.bishops & white_occ
    bb14[3]  = board.rooks   & white_occ
    bb14[4]  = board.queens  & white_occ
    bb14[5]  = board.kings   & white_occ
    bb14[6]  = board.pawns   & black_occ
    bb14[7]  = board.knights & black_occ
    bb14[8]  = board.bishops & black_occ
    bb14[9]  = board.rooks   & black_occ
    bb14[10] = board.queens  & black_occ
    bb14[11] = board.kings   & black_occ
    bb14[12] = white_occ
    bb14[13] = black_occ

    stm = 0 if board.turn else 1
    ep = board.ep_square if board.ep_square is not None else -1

    castle = 0
    if board.has_kingside_castling_rights(chess.WHITE):  castle |= 1
    if board.has_queenside_castling_rights(chess.WHITE): castle |= 2
    if board.has_kingside_castling_rights(chess.BLACK):  castle |= 4
    if board.has_queenside_castling_rights(chess.BLACK): castle |= 8

    lib.cboard_from_bitboards(ctypes.byref(cb), bb14, stm, ep, castle)
    return cb

def cmove_to_uci(cmove):
    """Decode CMove to UCI string."""
    from_sq = cmove & 0x3F
    to_sq = (cmove >> 6) & 0x3F
    promo = (cmove >> 12) & 0xF
    promo_chars = {0: '', 1: 'n', 2: 'b', 3: 'r', 4: 'q'}
    return chess.square_name(from_sq) + chess.square_name(to_sq) + promo_chars.get(promo, '')

def test_position(fen, label=""):
    """Compare C++ legal moves with python-chess for a given position."""
    board = chess.Board(fen)
    cb = sync_board(board)

    moves_buf = (ctypes.c_uint32 * 256)()
    count = lib.cboard_legal_moves(ctypes.byref(cb), moves_buf)

    cpp_moves = set()
    for i in range(count):
        cpp_moves.add(cmove_to_uci(moves_buf[i]))

    py_moves = set()
    for m in board.legal_moves:
        py_moves.add(m.uci())

    if cpp_moves == py_moves:
        print(f"  PASS [{count:3d} moves] {label}: {fen}")
        return True
    else:
        print(f"  FAIL [{count:3d} vs {len(py_moves):3d}] {label}: {fen}")
        missing = py_moves - cpp_moves
        extra = cpp_moves - py_moves
        if missing:
            print(f"    Missing from C++: {sorted(missing)}")
        if extra:
            print(f"    Extra in C++:     {sorted(extra)}")
        return False

def perft(board, depth):
    """Count total leaf nodes at given depth using python-chess."""
    if depth == 0:
        return 1
    count = 0
    for move in board.legal_moves:
        board.push(move)
        count += perft(board, depth - 1)
        board.pop()
    return count

def perft_cpp(cb_factory, board, depth):
    """Count leaf nodes using C++ movegen for move list, python-chess for push/pop."""
    if depth == 0:
        return 1

    cb = cb_factory(board)
    moves_buf = (ctypes.c_uint32 * 256)()
    count = lib.cboard_legal_moves(ctypes.byref(cb), moves_buf)
    n_moves = count

    total = 0
    for i in range(n_moves):
        uci = cmove_to_uci(moves_buf[i])
        move = chess.Move.from_uci(uci)
        board.push(move)
        total += perft_cpp(cb_factory, board, depth - 1)
        board.pop()
    return total

if __name__ == "__main__":
    print("=== Legal Move Generation Tests ===\n")

    positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting position"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "Kiwipete"),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Position 3"),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Position 4"),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", "Position 5 (promotions)"),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", "Position 6"),
        # EP positions
        ("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1", "After 1.e4 (EP available)"),
        ("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3", "EP capture available"),
        # Castling edge cases
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "Both sides can castle"),
        ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "Black to castle"),
        # Check positions
        ("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3", "White in check"),
        # Pinned pieces
        ("8/8/8/8/1k1Pp2q/8/8/4K3 w - d3 0 1", "EP with discovered check"),
    ]

    passed = 0
    failed = 0
    for fen, label in positions:
        if test_position(fen, label):
            passed += 1
        else:
            failed += 1

    print(f"\n{passed}/{passed+failed} positions passed.\n")

    # Perft comparison
    print("=== Perft Comparison (depth 3) ===\n")
    test_fens = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting pos"),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "Kiwipete"),
    ]

    for fen, label in test_fens:
        board = chess.Board(fen)
        py_count = perft(board, 3)
        cpp_count = perft_cpp(sync_board, board, 3)
        match = "PASS" if py_count == cpp_count else "FAIL"
        print(f"  {match} {label}: python-chess={py_count}, C++={cpp_count}")

    if failed > 0:
        sys.exit(1)
