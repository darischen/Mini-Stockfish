# distutils: language = c++
# distutils: libraries = nnue_inference chess_movegen
# distutils: library_dirs = nnue/build nnue/build/Release
# distutils: include_dirs = nnue

import os
from libc.string cimport memset
from libc.stdint cimport uint64_t, int64_t
from libc.stdlib cimport malloc, free
cdef bint USE_NNUE = False
cdef bint USE_TT = True

cpdef set_use_tt(bint flag):
    global USE_TT
    USE_TT = flag

import chess
from libc.math cimport INFINITY, log

cdef int[7] PIECE_VAL = [0, 100, 300, 310, 400, 900, 20000]

# HalfKP encoding constants (from halfkp.py)
cdef int NUM_NONKING = 10
cdef int PIECES_PER_KING = 64 * NUM_NONKING  # 640

# Map (color, piece_type) → offset
cdef dict _piece_to_idx = {
    (True,  chess.PAWN):   0,
    (True,  chess.KNIGHT): 1,
    (True,  chess.BISHOP): 2,
    (True,  chess.ROOK):   3,
    (True,  chess.QUEEN):  4,
    (False, chess.PAWN):   5,
    (False, chess.KNIGHT): 6,
    (False, chess.BISHOP): 7,
    (False, chess.ROOK):   8,
    (False, chess.QUEEN):  9,
}

cpdef tuple halfkp_indices_for_board(object board):
    """Compute HalfKP indices directly from board object (no FEN parsing).
    Returns (indices_view0, indices_view1) lists."""
    cdef list idx0 = []
    cdef list idx1 = []
    cdef int king_sq
    cdef int sq
    cdef object piece
    cdef int offset
    cdef int idx

    # View 0: white king perspective, View 1: black king perspective
    for view in (0, 1):
        king_sq = board.king(bool(view))
        for sq in range(64):
            piece = board.piece_at(sq)
            if piece and piece.piece_type != chess.KING:
                offset = _piece_to_idx[(piece.color, piece.piece_type)]
                idx = king_sq * PIECES_PER_KING + sq * NUM_NONKING + offset
                if view == 0:
                    idx0.append(idx)
                else:
                    idx1.append(idx)

    return idx0, idx1

cpdef bint verify_hash(uint64_t h_incremental, object board):
    """Debug: verify incremental hash matches full recompute."""
    cdef uint64_t h_full = compute_hash(board)
    if h_incremental != h_full:
        print(f"HASH MISMATCH! incremental={h_incremental}, full={h_full}, fen={board.fen()}")        
        return False
    return True

# ———————————————————————————Transposition Table———————————————————————————————————————————
cdef uint64_t zobrist_random[769]
cdef uint64_t zob_piece[12*64]
cdef uint64_t zob_side
cdef uint64_t zob_castle[4]
cdef uint64_t zob_ep[8]

def _init_zobrist_random():
    import random
    random.seed(0xC0FFEE)
    global zob_side

    # 1) Pieces (12 piece‐types * 64 squares)
    for i in range(12*64):
        zob_piece[i] = <uint64_t>random.getrandbits(64)

    # 2) Side to move
    zob_side = <uint64_t>random.getrandbits(64)

    # 3) Castling rights (K, Q, k, q)
    for i in range(4):
        zob_castle[i] = <uint64_t>random.getrandbits(64)

    # 4) En-passant file (a–h)
    for i in range(8):
        zob_ep[i] = <uint64_t>random.getrandbits(64)

cdef enum EntryFlag:
    EXACT = 0
    LOWERBOUND = 1
    UPPERBOUND = 2

cdef struct TTEntry:
    uint64_t key
    int           depth
    double        value
    unsigned char flag
    int           best_from   # from_square of best move (-1 = none)
    int           best_to     # to_square of best move

cdef TTEntry *tt_entries = NULL
cdef int       tt_size, tt_mask

cpdef init_tt(int size_pow2 = 1<<26):
    """
    Call once at module init (or from Python) to allocate the TT.
    """
    global tt_entries, tt_size, tt_mask
    tt_size = size_pow2
    tt_mask = size_pow2 - 1
    if tt_entries != NULL:
        free(tt_entries)
    tt_entries = <TTEntry*> malloc(tt_size * sizeof(TTEntry))
    # zero out everything
    memset(tt_entries, 0, tt_size * sizeof(TTEntry))

cpdef clear_tt():
    """
    Zero out all TT entries without reallocating.
    Call between iterative deepening iterations.
    """
    if tt_entries != NULL:
        memset(tt_entries, 0, tt_size * sizeof(TTEntry))

cdef inline int piece_index(int piece_type, bint is_white):
    return (piece_type-1)*2 + (0 if is_white else 1)

cpdef uint64_t compute_hash(object board):
    """
    Compute a full Zobrist hash from a chess.Board using our own
    zob_piece/zob_side/zob_castle/zob_ep tables, so incremental
    updates in the search are consistent with this root key.
    """
    cdef uint64_t h = 0
    cdef int pi, sq

    # 1) Pieces
    for sq, pc in board.piece_map().items():
        pi = piece_index(pc.piece_type, pc.color)
        h ^= zob_piece[pi * 64 + sq]

    # 2) Side to move (XOR if black to move)
    if not board.turn:  # board.turn == False means black
        h ^= zob_side

    # 3) Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        h ^= zob_castle[0]
    if board.has_queenside_castling_rights(chess.WHITE):
        h ^= zob_castle[1]
    if board.has_kingside_castling_rights(chess.BLACK):
        h ^= zob_castle[2]
    if board.has_queenside_castling_rights(chess.BLACK):
        h ^= zob_castle[3]

    # 4) En passant file
    if board.ep_square is not None:
        h ^= zob_ep[chess.square_file(board.ep_square)]

    return h

cpdef uint64_t update_hash_full(uint64_t h,
                                 object mv,
                                 object mover_piece,
                                 object captured_piece,
                                 bint old_castle_K, bint old_castle_Q,
                                 bint old_castle_k, bint old_castle_q,
                                 object old_ep_square,
                                 object board_after):
    """
    Incrementally update Zobrist hash. All info about pre-move state is passed in.
    `board_after` is the board AFTER push (used only for new castling/EP state).
    """
    cdef int from_sq = mv.from_square
    cdef int to_sq = mv.to_square
    cdef int pi, ep_cap_sq, pawn_pi2, promo_pi, rpi

    # 1) Flip side to move
    h ^= zob_side

    # 2) Remove old castling rights
    if old_castle_K:
        h ^= zob_castle[0]
    if old_castle_Q:
        h ^= zob_castle[1]
    if old_castle_k:
        h ^= zob_castle[2]
    if old_castle_q:
        h ^= zob_castle[3]

    # 3) Remove old EP
    if old_ep_square is not None:
        h ^= zob_ep[old_ep_square % 8]

    # 4) Remove captured piece (if any)
    if captured_piece is not None and captured_piece.piece_type != chess.KING:
        pi = piece_index(captured_piece.piece_type, captured_piece.color)
        # En passant: the captured pawn is NOT on to_sq
        if old_ep_square is not None and to_sq == old_ep_square and mover_piece.piece_type == 1:
            if mover_piece.color:  # white capturing
                ep_cap_sq = to_sq - 8
            else:
                ep_cap_sq = to_sq + 8
            h ^= zob_piece[pi * 64 + ep_cap_sq]
        else:
            h ^= zob_piece[pi * 64 + to_sq]

    # 5) Move the piece
    if mv.promotion is not None:
        # Remove pawn from from_sq
        pawn_pi2 = piece_index(1, mover_piece.color)
        h ^= zob_piece[pawn_pi2 * 64 + from_sq]
        # Add promoted piece to to_sq
        promoted = board_after.piece_at(to_sq)
        promo_pi = piece_index(promoted.piece_type, promoted.color)
        h ^= zob_piece[promo_pi * 64 + to_sq]
    else:
        pi = piece_index(mover_piece.piece_type, mover_piece.color)
        h ^= zob_piece[pi * 64 + from_sq]
        h ^= zob_piece[pi * 64 + to_sq]

    # 6) Castling rook movement
    if mover_piece.piece_type == chess.KING and abs(to_sq - from_sq) == 2:
        rpi = piece_index(chess.ROOK, mover_piece.color)
        if to_sq > from_sq:
            # Kingside
            h ^= zob_piece[rpi * 64 + (from_sq + 3)]
            h ^= zob_piece[rpi * 64 + (from_sq + 1)]
        else:
            # Queenside
            h ^= zob_piece[rpi * 64 + (from_sq - 4)]
            h ^= zob_piece[rpi * 64 + (from_sq - 1)]

    # 7) Add new castling rights
    if board_after.has_kingside_castling_rights(chess.WHITE):
        h ^= zob_castle[0]
    if board_after.has_queenside_castling_rights(chess.WHITE):
        h ^= zob_castle[1]
    if board_after.has_kingside_castling_rights(chess.BLACK):
        h ^= zob_castle[2]
    if board_after.has_queenside_castling_rights(chess.BLACK):
        h ^= zob_castle[3]

    # 8) Add new EP square
    if board_after.ep_square is not None:
        h ^= zob_ep[board_after.ep_square % 8]

    return h

cpdef uint64_t null_move_hash(uint64_t h,
                               bint old_castle_K, bint old_castle_Q,
                               bint old_castle_k, bint old_castle_q,
                               object old_ep_square,
                               object board_after):
    """
    Hash update for null move: flip side, adjust EP (castling doesn't change).
    """
    # Flip side
    h ^= zob_side

    # EP disappears after null move
    if old_ep_square is not None:
        h ^= zob_ep[old_ep_square % 8]
    # board_after.ep_square should be None after null move, but check anyway
    if board_after.ep_square is not None:
        h ^= zob_ep[board_after.ep_square % 8]

    return h

# ——————————————————————————————————————————————————————————————————————

cpdef set_use_nnue(bint flag):
     """
     Call this from Python before running search_root() to pick
     whether quiesce() uses NNUE or the old static evaluator.
     """
     global USE_NNUE
     USE_NNUE = flag

# ——————————————————————————————————————————————————————————————————————
#  1) import the Python buffer API
from cpython.buffer cimport (
    PyBUF_CONTIG_RO,
    PyBUF_FORMAT,
    PyObject_GetBuffer,
    PyBuffer_Release
)
#  2) import your C++ inference API
cdef extern from "nnue_inference.h":
    ctypedef void* NNUEHandle
    NNUEHandle nnue_create(const char* model_path) nogil
    void        nnue_destroy(NNUEHandle h) nogil
    double      nnue_eval(NNUEHandle h, const float* features, int length) nogil

cdef extern from *:
    """
    extern "C" double nnue_eval_halfkp(void* h,
                                       const int64_t* idx0, int len0,
                                       const int64_t* idx1, int len1);
    """
    double nnue_eval_halfkp(NNUEHandle h,
                             const int64_t* idx0, int len0,
                             const int64_t* idx1, int len1) nogil

#  3) C++ move generation API
cdef extern from "chess_movegen.h":
    ctypedef unsigned int CMove

    struct CBoard:
        uint64_t pieces[2][7]
        uint64_t occupied
        uint64_t occupied_co[2]
        int      side_to_move
        int      ep_square
        unsigned char castling

    void cboard_from_bitboards(CBoard* b, const uint64_t* bb14,
                                int stm, int ep, unsigned char castle) nogil
    int  cboard_legal_moves(const CBoard* b, CMove* moves) nogil
    void cboard_piece_array(const CBoard* b, unsigned char* out) nogil
    int  cboard_is_check(const CBoard* b) nogil
    int  cboard_is_capture(const CBoard* b, CMove mv) nogil
    void movegen_init() nogil

    int CMOVE_FROM(CMove m)
    int CMOVE_TO(CMove m)
    int CMOVE_PROMO(CMove m)
    int CMOVE_FLAGS(CMove m)

    int CMOVE_FLAG_CAPTURE
    int CMOVE_FLAG_EP
    int CMOVE_FLAG_CASTLE
    int CMOVE_FLAG_PROMOTION

# Re-declare the macros as inline cdef for Cython since macros aren't directly callable
cdef inline int _cmove_from(CMove m):
    return <int>(m & 0x3F)

cdef inline int _cmove_to(CMove m):
    return <int>((m >> 6) & 0x3F)

cdef inline int _cmove_promo(CMove m):
    return <int>((m >> 12) & 0xF)

cdef inline int _cmove_flags(CMove m):
    return <int>((m >> 16) & 0xF)

cdef CBoard _cboard
cdef uint64_t _bb14[14]

cdef void _sync_cboard(object board):
    """Sync python-chess board state into our CBoard struct."""
    cdef int stm, ep_int
    cdef unsigned char castle = 0

    # Piece bitboards: white pawns..kings, black pawns..kings
    _bb14[0]  = <uint64_t>board.pawns   & <uint64_t>board.occupied_co[1]  # white pawns
    _bb14[1]  = <uint64_t>board.knights  & <uint64_t>board.occupied_co[1]  # white knights
    _bb14[2]  = <uint64_t>board.bishops  & <uint64_t>board.occupied_co[1]  # white bishops
    _bb14[3]  = <uint64_t>board.rooks    & <uint64_t>board.occupied_co[1]  # white rooks
    _bb14[4]  = <uint64_t>board.queens   & <uint64_t>board.occupied_co[1]  # white queens
    _bb14[5]  = <uint64_t>board.kings    & <uint64_t>board.occupied_co[1]  # white kings
    _bb14[6]  = <uint64_t>board.pawns   & <uint64_t>board.occupied_co[0]  # black pawns
    _bb14[7]  = <uint64_t>board.knights  & <uint64_t>board.occupied_co[0]  # black knights
    _bb14[8]  = <uint64_t>board.bishops  & <uint64_t>board.occupied_co[0]  # black bishops
    _bb14[9]  = <uint64_t>board.rooks    & <uint64_t>board.occupied_co[0]  # black rooks
    _bb14[10] = <uint64_t>board.queens   & <uint64_t>board.occupied_co[0]  # black queens
    _bb14[11] = <uint64_t>board.kings    & <uint64_t>board.occupied_co[0]  # black kings
    _bb14[12] = <uint64_t>board.occupied_co[1]  # white occupied
    _bb14[13] = <uint64_t>board.occupied_co[0]  # black occupied

    stm = 0 if board.turn else 1
    ep_int = board.ep_square if board.ep_square is not None else -1

    cdef uint64_t cr = <uint64_t>(board.castling_rights)
    if cr & CASTLING_WK: castle |= 1
    if cr & CASTLING_WQ: castle |= 2
    if cr & CASTLING_BK: castle |= 4
    if cr & CASTLING_BQ: castle |= 8

    cboard_from_bitboards(&_cboard, _bb14, stm, ep_int, castle)

#  4) module‐level handle
cdef NNUEHandle _nnue = NULL

# —————————————————————————Probe/Store Helper————————————————————————————————
cdef inline double tt_probe(uint64_t key,
                            int required_depth,
                            double alpha,
                            double beta,
                            char *hit) nogil:
    cdef int idx = <int>(key & tt_mask)
    cdef TTEntry e = tt_entries[idx]
    hit[0] = 0
    if e.key != key or e.depth < required_depth:
        return 0
    if e.flag == EXACT:
        hit[0] = 1; return e.value
    if e.flag == LOWERBOUND and e.value >= beta:
        hit[0] = 1; return e.value
    if e.flag == UPPERBOUND and e.value <= alpha:
        hit[0] = 1; return e.value
    return 0

cdef inline void tt_store(uint64_t key,
                           int depth,
                           double value,
                           unsigned char flag,
                           int best_from = -1,
                           int best_to = -1) nogil:
    cdef int idx = <int>(key & tt_mask)
    cdef TTEntry *e = &tt_entries[idx]
    # Replace if: empty slot, deeper search, or same-depth EXACT upgrade
    if (e.key == 0
        or e.depth < depth
        or (e.depth == depth and flag == EXACT and e.flag != EXACT)):
        e.key   = key
        e.depth = depth
        e.value = value
        e.flag  = flag
        e.best_from = best_from
        e.best_to   = best_to

cdef inline object tt_get_best_move(uint64_t key):
    """Return the best move stored for this key, or None."""
    cdef int idx = <int>(key & tt_mask)
    cdef TTEntry e = tt_entries[idx]
    if e.key == key and e.best_from >= 0:
        return chess.Move(e.best_from, e.best_to)
    return None
# ——————————————————————————————————————————————————————————————————————

def init_nnue(model_path=None):
    """
    Load the TorchScript NNUE model into the native library.
    """
    global _nnue
    if _nnue:
        return
    if model_path is None:
        # __file__ here points to core_search.cp310-win_amd64.pyd
        base = os.path.dirname(__file__)
        model_path = os.path.join(base, "nnue", "halfkp_int8.pt")

    if not os.path.isfile(model_path):
        raise RuntimeError(f"NNUE model not found at {model_path!r}")
        
    _nnue = nnue_create(model_path.encode('utf-8'))
    if _nnue == NULL:
        raise RuntimeError(f"Could not load NNUE model from {model_path!r}")

cdef double _nnue_eval_view(float[:] buf) nogil:
    """
    Under nogil: call directly into your C++ API,
    passing the address of the first element.
    """
    cdef int length = buf.shape[0]
    return nnue_eval(_nnue, &buf[0], length)

cpdef double nnue_eval_py(object feat_buf):
    """
    Accept any object supporting the buffer protocol (e.g. numpy float32[771]).
    Calls into the native library under nogil.
    """
    cdef float[:] view
    cdef Py_buffer viewinfo

    # Torch‐tensor fast path: pull contiguous CPU numpy underlying buffer
    if hasattr(feat_buf, "data_ptr") and hasattr(feat_buf, "numel"):
        # Convert to CPU‐backed contiguous numpy array
        feat_buf = feat_buf.contiguous().cpu().numpy()

    # Generic buffer protocol fallback into a memoryview
    if PyObject_GetBuffer(feat_buf, &viewinfo, PyBUF_CONTIG_RO | PyBUF_FORMAT) != 0:
        raise ValueError("object does not support float32 buffer protocol")
    try:
        # Cast raw buffer to a Cython float[:] view
        view = <float[:viewinfo.len // sizeof(float)]> viewinfo.buf
        # Now call the pure‐C nogil entrypoint
        return _nnue_eval_view(view)
    finally:
        PyBuffer_Release(&viewinfo)
cdef double _nnue_eval_halfkp_view(int64_t[:] idx0, int64_t[:] idx1) nogil:
    """
    Under nogil: call directly into C++ HalfKP API.
    """
    cdef int len0 = idx0.shape[0]
    cdef int len1 = idx1.shape[0]
    return nnue_eval_halfkp(_nnue, &idx0[0], len0, &idx1[0], len1)

cpdef double nnue_eval_halfkp_py(object idx0_arr, object idx1_arr):
    """
    Accept numpy int64 arrays for HalfKP evaluation.
    """
    cdef int64_t[:] view0 = idx0_arr
    cdef int64_t[:] view1 = idx1_arr
    return _nnue_eval_halfkp_view(view0, view1)

# ——————————————————————————————————————————————————————————————————————

import cython
from chess import Board, Move
# from accumulator import Accumulator  # Avoid circular import; not used here
from libc.math cimport INFINITY

cdef double MATE_SCORE = 100000.0

# python-chess castling_rights bitmask: bit positions for rook squares
# WK = H1 = bit 7, WQ = A1 = bit 0, BK = H8 = bit 63, BQ = A8 = bit 56
cdef uint64_t CASTLING_WK = 1ULL << 7   # h1
cdef uint64_t CASTLING_WQ = 1ULL << 0   # a1
cdef uint64_t CASTLING_BK = 1ULL << 63  # h8
cdef uint64_t CASTLING_BQ = 1ULL << 56  # a8

cdef inline void _read_castling(object board,
                                  bint *ck, bint *cq, bint *ck2, bint *cq2):
    """Read all 4 castling rights from board.castling_rights bitmask in 1 attribute access."""
    cdef uint64_t cr = <uint64_t>(board.castling_rights)
    ck[0]  = (cr & CASTLING_WK) != 0
    cq[0]  = (cr & CASTLING_WQ) != 0
    ck2[0] = (cr & CASTLING_BK) != 0
    cq2[0] = (cr & CASTLING_BQ) != 0

# ——————————————————— History Heuristic Table ———————————————————————
# Indexed by [from_square][to_square]. Quiet moves that cause beta
# cutoffs get their score bumped by depth², so deeper cutoffs matter more.
cdef int history[64][64]

cpdef void clear_history():
    memset(history, 0, 64 * 64 * sizeof(int))
    clear_killers()
    clear_counters()

# ——————————————————— Killer Moves ———————————————————————————————————
# Track the last 2 moves that caused cutoffs at each depth
# Stores encoded move: (from_sq << 8) | to_sq
# killer[depth][0] = most recent cutoff move at this depth
# killer[depth][1] = second most recent cutoff move at this depth
cdef int killer[64][2]  # [depth][killer_slot]

cpdef void clear_killers():
    memset(killer, 0, 64 * 2 * sizeof(int))

# ——————————————————— Counter-Move Heuristic ———————————————————————
# counter[prev_from][prev_to] = best move we found after opponent played from→to
# Each cell stores [from_sq, to_sq] as a 16-bit encoded value
cdef int counter[64][64]  # Stores best move to play after opponent's move

cpdef void clear_counters():
    memset(counter, 0, 64 * 64 * sizeof(int))

# ——————————————————————————————————————————————————————————————————

# Python‐level counter variables
cdef public int nodes_evaluated = 0
cdef public int branches_pruned = 0
cdef public int tt_hits = 0
cdef public int tt_misses = 0

cpdef int get_nodes_evaluated():
    return nodes_evaluated

cpdef int get_branches_pruned():
    return branches_pruned

cpdef int get_tt_hits():
    return tt_hits

cpdef int get_tt_misses():
    return tt_misses

cpdef void reset_counters():
    global nodes_evaluated, branches_pruned
    nodes_evaluated = 0
    branches_pruned = 0

cpdef void reset_tt_counters():
    global tt_hits, tt_misses
    tt_hits = 0
    tt_misses = 0

# Per-depth node counters (for pruning rate measurement)
cdef int nodes_visited_per_depth[64]
cdef int nodes_pruned_per_depth[64]

cpdef void init_node_counters():
    """Initialize per-depth node counters."""
    for i in range(64):
        nodes_visited_per_depth[i] = 0
        nodes_pruned_per_depth[i] = 0

cpdef dict get_node_counters():
    """Return per-depth node counters as a dict."""
    return {
        'nodes_visited': list(nodes_visited_per_depth),
        'nodes_pruned': list(nodes_pruned_per_depth)
    }

cdef double static_eval(object board, object acc, str ai_color):
    """
    Simple material-based evaluation (fallback when not using NNUE).
    Returns score from perspective of ai_color.
    """
    cdef double score = 0.0
    cdef int piece_type

    for piece_type in range(1, 7):  # 1=pawn, 2=knight, 3=bishop, 4=rook, 5=queen, 6=king
        # White pieces
        white_count = len(board.pieces(piece_type, chess.WHITE))
        # Black pieces
        black_count = len(board.pieces(piece_type, chess.BLACK))

        score += PIECE_VAL[piece_type] * (white_count - black_count)

    # Return from ai_color's perspective
    if ai_color == "black":
        score = -score
    return score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double quiesce(object board,
                    object acc,
                    double alpha,
                    double beta,
                    str ai_color,
                    uint64_t key,
                    int depth = 0,
                    int tree_depth = 0):
    """
    Quiescence search with TT + incremental Zobrist hashing.
    `key` is the 64-bit hash for `board` before any moves here.
    `depth` is quiesce recursion depth.
    `tree_depth` is actual plies from root (used for correct mate scoring).
    """
    global tt_hits, tt_misses

    # Use C++ movegen for terminal check (avoids expensive python-chess legal_moves)
    cdef CMove _terminal_buf[256]
    _sync_cboard(board)
    if cboard_legal_moves(&_cboard, _terminal_buf) == 0:
        if cboard_is_check(&_cboard):
            return -(MATE_SCORE - tree_depth)
        else:
            return 0.0

    cdef char hit
    cdef double val

    # 0) probe TT
    val = tt_probe(key, 0, alpha, beta, &hit)
    if hit:
        tt_hits += 1
        return val
    tt_misses += 1

    # 1) stand-pat
    if USE_NNUE:
        val = acc.evaluate()
        # evaluate() returns from White's perspective; negamax needs side-to-move's
        if not board.turn:  # Black to move → flip
            val = -val
    else:
        val = static_eval(board, acc, ai_color)
        # static_eval returns from ai_color's perspective; flip if not side-to-move
        if board.turn != (ai_color == "white"):
            val = -val

    # 2) alpha/beta check on stand-pat
    if val >= beta:
        tt_store(key, 0, val, LOWERBOUND)
        return beta
    if val > alpha:
        alpha = val

    # 3) only consider captures
    cdef object mv, captured, mover
    cdef uint64_t next_key
    cdef double score
    cdef bint ck, cq, ck2, cq2
    cdef object old_ep
    for mv in order_moves(board):
        if not board.is_capture(mv):
            continue

        # For en passant, the captured pawn is not on to_sq
        if board.is_en_passant(mv):
            ep_cap_sq2 = mv.to_square + (-8 if board.turn else 8)
            captured = board.piece_at(ep_cap_sq2)
        else:
            captured = board.piece_at(mv.to_square)
        mover = board.piece_at(mv.from_square)

        # Save pre-move state for incremental hash (1 attr access vs 4 method calls)
        _read_castling(board, &ck, &cq, &ck2, &cq2)
        old_ep = board.ep_square

        # do the capture
        board.push(mv)
        acc.update(mv, captured, old_ep_square=old_ep)
        next_key = update_hash_full(key, mv, mover, captured,
                                     ck, cq, ck2, cq2, old_ep, board)

        # recurse with flipped colors and updated key
        score = -quiesce(board, acc, -beta, -alpha, ai_color, next_key, depth - 1, tree_depth + 1)

        board.pop()
        acc.rollback(mv, captured)

        # 4) cutoff?
        if score >= beta:
            tt_store(key, 0, score, LOWERBOUND)
            return beta
        if score > alpha:
            alpha = score

    # 5) store exact and return
    tt_store(key, 0, alpha, EXACT)
    return alpha

cdef bint _has_non_pawn_material(object board, bint side):
    """Check if side has any non-pawn, non-king material (for null move safety)."""
    for pt in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
        if board.pieces(pt, side):
            return True
    return False

# Null move pruning depth reduction
cdef int NMP_REDUCTION = 2

# LMR: how many full-depth moves before reducing
cdef int LMR_FULL_MOVES = 2
# LMR: minimum depth to apply reduction
cdef int LMR_MIN_DEPTH = 2

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double minimax(object board,
                     object acc,
                     int depth,
                     double alpha,
                     double beta,
                     str ai_color,
                     uint64_t key,
                     int required_depth,
                     object prev_move = None):
    """
    Negamax with alpha-beta, TT, null move pruning, and late move reduction.
    - `key` is the current zobrist hash for `board`.
    - `depth` is the remaining search depth at this node.
    - `required_depth` is the root iteration depth, passed unchanged to
      all children. TT entries are stored AND probed with this value,
      ensuring each iteration fully re-searches the tree.
    - `prev_move` is the move that led to this position (for counter-move heuristic).
      Within a single iteration, TT reuse happens via matching keys
      (same position reached via different move orders).
    """
    global nodes_evaluated, branches_pruned, tt_hits, tt_misses
    cdef double value, child, cached, null_score
    cdef object mv, captured
    cdef uint64_t next_key, null_key
    cdef char hit
    cdef int moves_searched, reduced_depth, actual_depth, reduction, nmp_R
    cdef bint is_capture, gives_check, is_promotion
    cdef bint in_check
    cdef double lmr_r

    # Count nodes
    nodes_evaluated += 1
    actual_depth = required_depth - depth
    if actual_depth < 64:
        nodes_visited_per_depth[actual_depth] += 1

    # 1) TT probe — use `depth` (remaining search depth).
    #    TT is cleared between iterations so no cross-iteration pollution.
    cdef object tt_move = None
    if USE_TT:
        cached = tt_probe(key, depth, alpha, beta, &hit)
        if hit:
            tt_hits += 1
            return cached
        tt_misses += 1
        # Even if depth was insufficient for a cutoff, grab the best move
        # for move ordering — it's still the best move found at this position.
        tt_move = tt_get_best_move(key)

    # 2) Terminal — use C++ movegen for fast legal move check
    _sync_cboard(board)
    cdef CMove _term_buf2[256]
    if cboard_legal_moves(&_cboard, _term_buf2) == 0:
        if cboard_is_check(&_cboard):
            actual_ply_count = required_depth - depth
            return -(MATE_SCORE - actual_ply_count)
        else:
            return 0.0

    # 3) Leaf → quiescence
    if depth == 0:
        child = quiesce(board, acc, alpha, beta, ai_color, key, 0, required_depth)
        if USE_TT:
            tt_store(key, depth, child, EXACT)
        return child

    in_check = board.is_check()

    cdef bint ck, cq, ck2, cq2
    cdef object old_ep, mover
    cdef object best_mv = None

    # ——— Null Move Pruning (Ethereal-style dynamic reduction) ———
    # Skip if: in check, depth too shallow, no non-pawn material (zugzwang risk),
    # or beta is infinity (PV node with wide-open window — null move can't prune reliably)
    nmp_R = 3 + depth // 4  # Dynamic reduction: deeper search → more reduction
    if (not in_check
        and depth >= nmp_R + 1
        and beta < INFINITY
        and _has_non_pawn_material(board, board.turn)):
        # Save pre-move state
        _read_castling(board, &ck, &cq, &ck2, &cq2)
        old_ep = board.ep_square

        # Play a "null move" (pass turn) and search with reduced depth
        null_mv = chess.Move.null()
        board.push(null_mv)
        acc.update(null_mv, None, old_ep_square=old_ep)
        null_key = null_move_hash(key, ck, cq, ck2, cq2, old_ep, board)
        null_score = -minimax(board, acc,
                              depth - 1 - nmp_R,
                              -beta, -beta + 1,
                              ai_color,
                              null_key,
                              required_depth,
                              prev_move)
        board.pop()
        acc.rollback(null_mv, None)

        if null_score >= beta:
            branches_pruned += 1
            if actual_depth < 64:
                nodes_pruned_per_depth[actual_depth] += 1
            return beta

    # 4) Negamax loop with Late Move Reduction
    value = -INFINITY
    moves_searched = 0
    # Use required_depth - depth to get the actual search depth for killer/counter indexing
    search_depth = required_depth - depth
    for mv in order_moves(board, tt_move, search_depth, prev_move):
        is_capture = board.is_capture(mv)

        # For en passant, the captured pawn is not on to_sq
        if is_capture and board.is_en_passant(mv):
            ep_cap_sq3 = mv.to_square + (-8 if board.turn else 8)
            captured = board.piece_at(ep_cap_sq3)
        else:
            captured = board.piece_at(mv.to_square)
        mover = board.piece_at(mv.from_square)

        # Save pre-move state for incremental hash
        _read_castling(board, &ck, &cq, &ck2, &cq2)
        old_ep = board.ep_square

        board.push(mv)
        acc.update(mv, captured, old_ep_square=old_ep)
        next_key = update_hash_full(key, mv, mover, captured,
                                     ck, cq, ck2, cq2, old_ep, board)

        is_promotion = mv.promotion is not None
        gives_check = board.is_check()

        # ——— Late Move Reduction (Ethereal-style log formula) ———
        # After searching the first few moves at full depth, reduce later
        # quiet moves (non-captures, non-checks, non-promotions).
        if (moves_searched >= LMR_FULL_MOVES
            and depth >= LMR_MIN_DEPTH
            and not is_capture
            and not gives_check
            and not is_promotion
            and not in_check):
            # Ethereal-style: r = 0.78 + ln(depth) * ln(moves_searched) / 2.47
            lmr_r = 0.78 + log(<double>depth) * log(<double>moves_searched) / 2.47
            reduction = <int>lmr_r
            reduced_depth = depth - 1 - reduction
            if reduced_depth < 1:
                reduced_depth = 1
            child = -minimax(board, acc,
                             reduced_depth,
                             -beta, -alpha,
                             ai_color,
                             next_key,
                             required_depth,
                             mv)
            # If reduced search beats alpha, re-search at full depth
            if child > alpha:
                child = -minimax(board, acc,
                                 depth - 1,
                                 -beta, -alpha,
                                 ai_color,
                                 next_key,
                                 required_depth,
                                 mv)
        else:
            # Full depth search for important moves
            child = -minimax(board, acc,
                             depth-1,
                             -beta, -alpha,
                             ai_color,
                             next_key,
                             required_depth,
                             mv)

        board.pop()
        acc.rollback(mv, captured)

        moves_searched += 1

        if child > value:
            value = child
            best_mv = mv
        if value > alpha:
            alpha = value

        if alpha >= beta:
            branches_pruned += 1
            if actual_depth < 64:
                nodes_pruned_per_depth[actual_depth] += 1
            # Update history, killers, and counter-moves for quiet cutoff moves
            if not is_capture:
                # History: reward quiet moves that cause cutoffs
                history[mv.from_square][mv.to_square] += depth * depth

                # Killer moves: track which moves caused cutoffs at this depth
                if search_depth < 64:
                    # Move killer[depth][1] → killer[depth][0], and add new killer
                    killer[search_depth][1] = killer[search_depth][0]
                    killer[search_depth][0] = (mv.from_square << 8) | mv.to_square

                # Counter-move: update what move works well after opponent's prev_move
                if prev_move is not None:
                    prev_from = prev_move.from_square
                    prev_to = prev_move.to_square
                    if 0 <= prev_from < 64 and 0 <= prev_to < 64:
                        counter[prev_from][prev_to] = (mv.from_square << 8) | mv.to_square

            if USE_TT:
                tt_store(key, depth, value, LOWERBOUND,
                         mv.from_square, mv.to_square)
            return value

    # 5) If no moves were searched (all pruned), fall back to static/quiesce eval
    if moves_searched == 0:
        child = quiesce(board, acc, alpha, beta, ai_color, key)
        if USE_TT:
            tt_store(key, depth, child, EXACT)
        return child

    # 6) store exact with best move and return
    if USE_TT:
        if best_mv is not None:
            tt_store(key, depth, value, EXACT,
                     best_mv.from_square, best_mv.to_square)
        else:
            tt_store(key, depth, value, EXACT)
    return value

cdef int PROMO_TO_CHESS[5]
PROMO_TO_CHESS[0] = 0  # no promotion
PROMO_TO_CHESS[1] = 2  # knight (CMove=1 → chess.KNIGHT=2)
PROMO_TO_CHESS[2] = 3  # bishop (CMove=2 → chess.BISHOP=3)
PROMO_TO_CHESS[3] = 4  # rook   (CMove=3 → chess.ROOK=4)
PROMO_TO_CHESS[4] = 5  # queen  (CMove=4 → chess.QUEEN=5)

cdef list order_moves(object board, object tt_move = None, int depth = 0, object prev_move = None):
    """
    Generates and scores all legal moves for `board` using C++ movegen.
    Returns a Python list of chess.Move in descending score order (best first).

    Priority tiers:
      1. TT best move              (score += 10_000_000)
      2. Killer moves              (score += 5_000_000)
      3. Counter-move              (score += 1_000_000)
      4. Captures by MVV/LVA       (score = 2000*victim - attacker)
      5. Quiet moves by history    (score = history[from][to])
    """
    cdef CMove c_moves[256]
    cdef unsigned char pieces[64]
    cdef int n_moves, i
    cdef int from_sq, to_sq, flags, promo
    cdef int a_pt, v_pt, a_val, v_val, score
    cdef int tt_from = -1, tt_to = -1, tt_promo = 0
    cdef int killer1_from = -1, killer1_to = -1, killer2_from = -1, killer2_to = -1
    cdef int counter_move = -1, prev_from = -1, prev_to = -1
    cdef list scored = []
    cdef int move_index_count = 0

    # Sync python-chess board → CBoard and generate moves + piece array in C++
    _sync_cboard(board)
    n_moves = cboard_legal_moves(&_cboard, c_moves)
    cboard_piece_array(&_cboard, pieces)

    # Pre-extract TT move fields for fast comparison (avoid Python == per move)
    if tt_move is not None:
        tt_from = tt_move.from_square
        tt_to = tt_move.to_square
        tt_promo = tt_move.promotion if tt_move.promotion is not None else 0

    # Pre-extract killer moves at this depth
    if 0 <= depth < 64:
        killer1_from = (killer[depth][0] >> 8) & 0xFF
        killer1_to = killer[depth][0] & 0xFF
        killer2_from = (killer[depth][1] >> 8) & 0xFF
        killer2_to = killer[depth][1] & 0xFF

    # Pre-extract previous move for counter-move heuristic
    if prev_move is not None:
        prev_from = prev_move.from_square
        prev_to = prev_move.to_square
        if 0 <= prev_from < 64 and 0 <= prev_to < 64:
            counter_move = counter[prev_from][prev_to]

    for i in range(n_moves):
        from_sq = _cmove_from(c_moves[i])
        to_sq   = _cmove_to(c_moves[i])
        flags   = _cmove_flags(c_moves[i])
        promo   = _cmove_promo(c_moves[i])

        # Attacker piece type from piece array: low 4 bits = piece_type
        a_pt = pieces[from_sq] & 0xF
        a_val = PIECE_VAL[a_pt] if a_pt else 0

        # Victim scoring — use flags from C++ (no Python method calls)
        v_val = 0
        if flags & 1:  # CMOVE_FLAG_CAPTURE
            if flags & 2:  # CMOVE_FLAG_EP
                # En passant: captured pawn is behind target square
                v_val = PIECE_VAL[1]  # pawn value
            else:
                v_pt = pieces[to_sq] & 0xF
                v_val = PIECE_VAL[v_pt] if v_pt else 0

        if v_val > 0:
            score = 2000 * v_val - a_val
        else:
            # Quiet move: start with history heuristic
            score = history[from_sq][to_sq]

            # Killer move bonus: moves that caused cutoffs at this depth
            if (killer1_from == from_sq and killer1_to == to_sq):
                score += 5000000
            elif (killer2_from == from_sq and killer2_to == to_sq):
                score += 4900000

            # Counter-move bonus: responds well to opponent's last move
            if counter_move != -1:
                counter_from = (counter_move >> 8) & 0xFF
                counter_to = counter_move & 0xFF
                if (counter_from == from_sq and counter_to == to_sq):
                    score += 1000000

        # Promotion bonus
        if flags & 8:  # CMOVE_FLAG_PROMOTION
            score += PIECE_VAL[PROMO_TO_CHESS[promo]]

        # TT best move gets absolute priority
        if (tt_from == from_sq and tt_to == to_sq
            and (tt_promo == 0 or tt_promo == PROMO_TO_CHESS[promo])):
            score += 10000000

        scored.append((c_moves[i], score))

    # Sort by score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Convert CMove → chess.Move
    cdef list result = []
    cdef CMove cm
    cdef int rf, rt, rp, rfl
    for cm, _ in scored:
        rf = _cmove_from(cm)
        rt = _cmove_to(cm)
        rp = _cmove_promo(cm)
        move_obj = None
        if rp:
            move_obj = chess.Move(rf, rt, promotion=PROMO_TO_CHESS[rp])
        else:
            move_obj = chess.Move(rf, rt)
        result.append(move_obj)
    return result

# Allocate a 4M-entry table by default
_init_zobrist_random()
init_tt(1<<26)
movegen_init()
