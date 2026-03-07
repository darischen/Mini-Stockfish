# distutils: language = c++
# distutils: libraries = nnue_inference
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
from libc.math cimport INFINITY

cdef int[7] PIECE_VAL = [0, 100, 300, 310, 400, 900, 20000]

cpdef bint verify_hash(uint64_t h_incremental, object board):
    """Debug: verify incremental hash matches full recompute."""
    cdef uint64_t h_full = compute_hash(board)
    if h_incremental != h_full:
        print(f"HASH MISMATCH! incremental={h_incremental}, full={h_full}, fen={board.fen()}")        
        return False
    return True

cpdef int see(object board, object mv):
    cdef object vic, att
    cdef int v_pt, a_pt

    # only consider captures
    if not board.is_capture(mv):
        return -100000

    vic = board.piece_at(mv.to_square)
    if vic is None:
        # weird edge‐case: nothing to capture
        return -100000

    att = board.piece_at(mv.from_square)
    if att is None:
        # also weird: no attacker
        return -100000

    v_pt = vic.piece_type
    a_pt = att.piece_type

    # positive if we net gain, negative if we net lose
    return PIECE_VAL[v_pt] - PIECE_VAL[a_pt]

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

cdef TTEntry *tt_entries = NULL
cdef int       tt_size, tt_mask

cpdef init_tt(int size_pow2 = 1<<28):
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

#  3) module‐level handle
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
                           unsigned char flag) nogil:
    cdef int idx = <int>(key & tt_mask)
    cdef TTEntry *e = &tt_entries[idx]
    # only replace if deeper, or same-depth EXACT overrides
    if e.depth < depth or (e.depth == depth and flag == EXACT and e.flag != EXACT):
        e.key   = key
        e.depth = depth
        e.value = value
        e.flag  = flag
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
from accumulator import Accumulator
from libc.math cimport INFINITY

cdef double MATE_SCORE = 100000.0

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

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double static_eval(object board, object acc, str ai_color):
    cdef dict pm = board.piece_map()
    cdef double score = 0.0
    cdef bint piece_is_white, ai_is_white, us, them
    cdef int pt, mob, sq, file, rank, f, r, ep_file, ep_rank
    cdef int king_sq, king_file, king_rank, effective_rank
    cdef int pawns_on_file, bishop_count
    cdef double val, shield
    cdef list our_pawns, enemy_pawns
    cdef bint is_passed, has_neighbor, has_our_pawn, has_enemy_pawn

    ai_is_white = (ai_color == "white")
    us = ai_is_white
    them = not us

    # --- Material ---
    for sq, p in pm.items():
        piece_is_white = p.color
        pt = p.piece_type
        if pt == 1:
            val = 100.0
        elif pt == 2:
            val = 300.0
        elif pt == 3:
            val = 310.0
        elif pt == 4:
            val = 400.0
        elif pt == 5:
            val = 900.0
        else:
            val = 20000.0
        score += val if piece_is_white == ai_is_white else -val

    # --- Mobility ---
    mob = board.legal_moves.count()
    score += mob * 10 if board.turn == ai_is_white else -mob * 10

    # --- Bishop pair ---
    bishop_count = 0
    for sq, p in pm.items():
        if p.piece_type == 3 and p.color == us:  # BISHOP = 3
            bishop_count += 1
    if bishop_count >= 2:
        score += 50

    # --- Passed pawns ---
    cdef int[8] passed_bonus = [0, 10, 20, 40, 70, 120, 200, 0]
    our_pawns = list(board.pieces(1, us))  # PAWN = 1
    enemy_pawns = list(board.pieces(1, them))

    for sq in our_pawns:
        file = sq % 8
        rank = sq // 8
        is_passed = True
        for ep in enemy_pawns:
            ep_file = ep % 8
            ep_rank = ep // 8
            if abs(ep_file - file) <= 1:
                if us and ep_rank > rank:
                    is_passed = False
                    break
                elif not us and ep_rank < rank:
                    is_passed = False
                    break
        if is_passed:
            effective_rank = rank if us else 7 - rank
            score += passed_bonus[effective_rank]

    # --- Doubled pawns penalty ---
    for f in range(8):
        pawns_on_file = 0
        for p in our_pawns:
            if p % 8 == f:
                pawns_on_file += 1
        if pawns_on_file > 1:
            score -= 20 * (pawns_on_file - 1)

    # --- Isolated pawns penalty ---
    for sq in our_pawns:
        file = sq % 8
        has_neighbor = False
        for p in our_pawns:
            if p != sq and abs(p % 8 - file) == 1:
                has_neighbor = True
                break
        if not has_neighbor:
            score -= 15

    # --- Rook on open/semi-open file ---
    for sq in board.pieces(4, us):  # ROOK = 4
        file = sq % 8
        has_our_pawn = False
        has_enemy_pawn = False
        for p in our_pawns:
            if p % 8 == file:
                has_our_pawn = True
                break
        for p in enemy_pawns:
            if p % 8 == file:
                has_enemy_pawn = True
                break
        if not has_our_pawn and not has_enemy_pawn:
            score += 40  # Open file
        elif not has_our_pawn:
            score += 20  # Semi-open file

    # --- Rook on 7th rank ---
    for sq in board.pieces(4, us):  # ROOK = 4
        rank = sq // 8
        if (us and rank == 6) or (not us and rank == 1):
            score += 50

    # --- King safety (pawn shield) ---
    king_sq_obj = board.king(us)
    if king_sq_obj is not None:
        king_sq = king_sq_obj
        king_file = king_sq % 8
        king_rank = king_sq // 8
        shield = 0.0
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            if us:
                # White: check ranks above king
                for r in [king_rank + 1, king_rank + 2]:
                    if 0 <= r < 8:
                        sq = r * 8 + f
                        p = board.piece_at(sq)
                        if p is not None and p.piece_type == 1 and p.color == us:
                            shield += 15 if r == king_rank + 1 else 10
                            break
            else:
                # Black: check ranks below king
                for r in [king_rank - 1, king_rank - 2]:
                    if 0 <= r < 8:
                        sq = r * 8 + f
                        p = board.piece_at(sq)
                        if p is not None and p.piece_type == 1 and p.color == us:
                            shield += 15 if r == king_rank - 1 else 10
                            break
        score += shield

        # Penalty for open files near king
        for f in range(max(0, king_file - 1), min(8, king_file + 2)):
            has_our_pawn = False
            has_enemy_pawn = False
            for p in our_pawns:
                if p % 8 == f:
                    has_our_pawn = True
                    break
            for p in enemy_pawns:
                if p % 8 == f:
                    has_enemy_pawn = True
                    break
            if not has_our_pawn and not has_enemy_pawn:
                score -= 25
            elif not has_our_pawn:
                score -= 15

    # --- Center control ---
    cdef list center = [28, 27, 36, 35]  # e4, d4, e5, d5
    for sq in center:
        if board.is_attacked_by(us, sq):
            score += 10
        p = board.piece_at(sq)
        if p is not None and p.color == us:
            if p.piece_type == 1:  # Pawn
                score += 20
            elif p.piece_type in [2, 3]:  # Knight, Bishop
                score += 15

    # --- Hanging piece penalty ---
    for sq, p in pm.items():
        if p.color == us:
            if board.is_attacked_by(them, sq) and not board.is_attacked_by(us, sq):
                score -= PIECE_VAL[p.piece_type] * 0.5

    return score

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double quiesce(object board,
                    object acc,
                    double alpha,
                    double beta,
                    str ai_color,
                    uint64_t key):
    """
    Quiescence search with TT + incremental Zobrist hashing.
    `key` is the 64-bit hash for `board` before any moves here.
    """
    if not board.legal_moves:
        if board.is_check():
            return -MATE_SCORE
        else:
            return 0.0

    cdef char hit
    cdef double val

    # 0) probe TT
    val = tt_probe(key, 0, alpha, beta, &hit)
    if hit:
        return val

    # 1) stand-pat
    if USE_NNUE:
        val = nnue_eval_halfkp_py(acc.idx0, acc.idx1)
        # NNUE returns from White's perspective; negamax needs side-to-move's
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

        # Save pre-move state for incremental hash
        ck = board.has_kingside_castling_rights(chess.WHITE)
        cq = board.has_queenside_castling_rights(chess.WHITE)
        ck2 = board.has_kingside_castling_rights(chess.BLACK)
        cq2 = board.has_queenside_castling_rights(chess.BLACK)
        old_ep = board.ep_square

        # do the capture
        board.push(mv)
        acc.update(mv, captured)
        next_key = update_hash_full(key, mv, mover, captured,
                                     ck, cq, ck2, cq2, old_ep, board)

        if not verify_hash(next_key, board):
            next_key = compute_hash(board)

        # recurse with flipped colors and updated key
        score = -quiesce(board, acc, -beta, -alpha, ai_color, next_key)

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
cdef int LMR_FULL_MOVES = 4
# LMR: minimum depth to apply reduction
cdef int LMR_MIN_DEPTH = 3

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double minimax(object board,
                     object acc,
                     int depth,
                     double alpha,
                     double beta,
                     str ai_color,
                     uint64_t key,
                     int required_depth):
    """
    Negamax with alpha-beta, TT, null move pruning, and late move reduction.
    - `key` is the current zobrist hash for `board`.
    - `depth` is the remaining search depth at this node.
    - `required_depth` is the root iteration depth, passed unchanged to
      all children. TT entries are stored AND probed with this value,
      ensuring each iteration fully re-searches the tree.
      Within a single iteration, TT reuse happens via matching keys
      (same position reached via different move orders).
    """
    global nodes_evaluated, branches_pruned, tt_hits, tt_misses
    cdef double value, child, cached, null_score
    cdef object mv, captured
    cdef uint64_t next_key, null_key
    cdef char hit
    cdef int moves_searched, reduced_depth
    cdef bint is_capture, gives_check, is_promotion
    cdef bint in_check

    # Count nodes
    nodes_evaluated += 1

    # 1) TT probe — use `depth` (remaining search depth).
    #    TT is cleared between iterations so no cross-iteration pollution.
    if USE_TT:
        cached = tt_probe(key, depth, alpha, beta, &hit)
        if hit:
            tt_hits += 1
            return cached
        tt_misses += 1

    # 2) Terminal — only check checkmate/stalemate, not draw claims
    #    (draw claims can give false positives with fresh board copies)
    if not board.legal_moves:
        if board.is_check():
            return -MATE_SCORE + depth
        else:
            return 0.0

    # 3) Leaf → quiescence
    if depth == 0:
        child = quiesce(board, acc, alpha, beta, ai_color, key)
        if USE_TT:
            tt_store(key, depth, child, EXACT)
        return child

    in_check = board.is_check()

    cdef bint ck, cq, ck2, cq2
    cdef object old_ep, mover

    # ——— Null Move Pruning ———
    # Skip if: in check, depth too shallow, no non-pawn material (zugzwang risk),
    # or beta is infinity (PV node with wide-open window — null move can't prune reliably)
    if (not in_check
        and depth >= NMP_REDUCTION + 1
        and beta < INFINITY
        and _has_non_pawn_material(board, board.turn)):
        # Save pre-move state
        ck = board.has_kingside_castling_rights(chess.WHITE)
        cq = board.has_queenside_castling_rights(chess.WHITE)
        ck2 = board.has_kingside_castling_rights(chess.BLACK)
        cq2 = board.has_queenside_castling_rights(chess.BLACK)
        old_ep = board.ep_square

        # Play a "null move" (pass turn) and search with reduced depth
        null_mv = chess.Move.null()
        board.push(null_mv)
        acc.update(null_mv, None)
        null_key = null_move_hash(key, ck, cq, ck2, cq2, old_ep, board)
        if not verify_hash(null_key, board):
            null_key = compute_hash(board)
        null_score = -minimax(board, acc,
                              depth - 1 - NMP_REDUCTION,
                              -beta, -beta + 1,
                              ai_color,
                              null_key,
                              required_depth)
        board.pop()
        acc.rollback(null_mv, None)

        if null_score >= beta:
            branches_pruned += 1
            return beta

    # 4) Negamax loop with Late Move Reduction
    value = -INFINITY
    moves_searched = 0
    for mv in order_moves(board):
        is_capture = board.is_capture(mv)

        # For en passant, the captured pawn is not on to_sq
        if is_capture and board.is_en_passant(mv):
            ep_cap_sq3 = mv.to_square + (-8 if board.turn else 8)
            captured = board.piece_at(ep_cap_sq3)
        else:
            captured = board.piece_at(mv.to_square)
        mover = board.piece_at(mv.from_square)

        # Save pre-move state for incremental hash
        ck = board.has_kingside_castling_rights(chess.WHITE)
        cq = board.has_queenside_castling_rights(chess.WHITE)
        ck2 = board.has_kingside_castling_rights(chess.BLACK)
        cq2 = board.has_queenside_castling_rights(chess.BLACK)
        old_ep = board.ep_square

        board.push(mv)
        acc.update(mv, captured)
        next_key = update_hash_full(key, mv, mover, captured,
                                     ck, cq, ck2, cq2, old_ep, board)

        if not verify_hash(next_key, board):
            # Hash is wrong, fall back to full recompute
            next_key = compute_hash(board)

        is_promotion = mv.promotion is not None
        gives_check = board.is_check()

        # ——— Late Move Reduction ———
        # After searching the first few moves at full depth, reduce later
        # quiet moves (non-captures, non-checks, non-promotions).
        if (moves_searched >= LMR_FULL_MOVES
            and depth >= LMR_MIN_DEPTH
            and not is_capture
            and not gives_check
            and not is_promotion
            and not in_check):
            # Search with reduced depth first
            reduced_depth = depth - 2  # reduce by 1 extra ply
            if reduced_depth < 1:
                reduced_depth = 1
            child = -minimax(board, acc,
                             reduced_depth - 1,
                             -beta, -alpha,
                             ai_color,
                             next_key,
                             required_depth)
            # If reduced search beats alpha, re-search at full depth
            if child > alpha:
                child = -minimax(board, acc,
                                 depth - 1,
                                 -beta, -alpha,
                                 ai_color,
                                 next_key,
                                 required_depth)
        else:
            # Full depth search for important moves
            child = -minimax(board, acc,
                             depth-1,
                             -beta, -alpha,
                             ai_color,
                             next_key,
                             required_depth)

        board.pop()
        acc.rollback(mv, captured)

        moves_searched += 1

        if child > value:
            value = child
        if value > alpha:
            alpha = value

        if alpha >= beta:
            branches_pruned += 1
            if USE_TT:
                tt_store(key, depth, value, LOWERBOUND)
            return value

    # 5) If no moves were searched (all pruned), fall back to static/quiesce eval
    if moves_searched == 0:
        child = quiesce(board, acc, alpha, beta, ai_color, key)
        if USE_TT:
            tt_store(key, depth, child, EXACT)
        return child

    # 6) store exact and return
    if USE_TT:
        tt_store(key, depth, value, EXACT)
    return value

cdef list order_moves(object board):
    """
    Generates and scores all legal moves for `board`.
    Returns a Python list of moves in descending score order (best first).
    No hard pruning — move ordering only. The search handles pruning via alpha-beta.
    """
    cdef int us       = board.turn
    cdef int them     = not us
    cdef object mv, attacker, victim
    cdef int from_sq, to_sq, a_val, v_val, score, ep_sq
    cdef uint64_t pawn_attack_mask = 0
    cdef list scored = []

    # 1) Precompute pawn attacks for scoring (skip full opp_attacks — too expensive)
    for sq in board.pieces(chess.PAWN, them):
        pawn_attack_mask |= board.attacks_mask(sq)

    # 2) Loop & score (no hard pruning — let the search decide)
    for mv in board.legal_moves:
        from_sq = mv.from_square
        to_sq   = mv.to_square

        # Attacker value
        attacker = board.piece_at(from_sq)
        a_val = PIECE_VAL[attacker.piece_type] if attacker else 0

        # Victim value
        if board.is_capture(mv):
            if board.is_en_passant(mv):
                ep_sq = mv.to_square + (8 if us else -8)
                victim = board.piece_at(ep_sq)
            else:
                victim = board.piece_at(to_sq)
            v_val = PIECE_VAL[victim.piece_type] if victim else 0
        else:
            v_val = 0

        # MVV/LVA core
        score = 2000 * v_val - a_val

        # Promotion bonus
        if mv.promotion is not None:
            score += PIECE_VAL[mv.promotion]

        # Pawn-attack penalty
        if (pawn_attack_mask >> to_sq) & 1:
            score -= 10 * a_val

        scored.append((mv, score))

    # 3) Final sort & return
    scored.sort(key=lambda x: x[1], reverse=True)
    return [m for m, _ in scored]

# Allocate a 4M-entry table by default
_init_zobrist_random()
init_tt(1<<28)