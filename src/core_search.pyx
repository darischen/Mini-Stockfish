# distutils: language = c++
# distutils: libraries = nnue_inference
# distutils: library_dirs = nnue/build
# distutils: include_dirs = nnue

import os
from libc.string cimport memset
from libc.stdint cimport uint64_t
from libc.stdlib cimport malloc, free
from chess.polyglot import zobrist_hash
cdef bint USE_NNUE = False

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

# then, at module init
_init_zobrist_random()

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

cpdef init_tt(int size_pow2 = 1<<20):
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

cdef inline int piece_index(int piece_type, bint is_white):
    return (piece_type-1)*2 + (0 if is_white else 1)
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
                           unsigned char flag) except * nogil:
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
        model_path = os.path.join(base, "nnue", "7.7e-3.pt")

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


# ——————————————————————————————————————————————————————————————————————
import cython
from chess import Board, Move
from accumulator import Accumulator
from libc.math cimport INFINITY

cdef double MATE_SCORE = 100000.0

# Python‐level counter variables
cdef public int nodes_evaluated = 0
cdef public int branches_pruned = 0

cpdef int get_nodes_evaluated():
    return nodes_evaluated

cpdef int get_branches_pruned():
    return branches_pruned

cpdef void reset_counters():
    global nodes_evaluated, branches_pruned
    nodes_evaluated = 0
    branches_pruned = 0

cdef class SearchResult:
    cdef public double value
    cdef public object uci
    def __init__(self, double value, object uci):
        self.value = value
        self.uci = uci

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double static_eval(object board, object acc, str ai_color):
    cdef dict pm = board.piece_map()
    cdef double score = 0.0
    cdef bint piece_is_white, ai_is_white
    cdef int pt, mob
    cdef double val

    ai_is_white = (ai_color == "white")
    for _, p in pm.items():
        piece_is_white = p.color
        pt = p.piece_type
        if pt == 1:
            val = 100.0
        elif pt == 2:
            val = 300.0
        elif pt == 3:
            val = 310.0
        elif pt == 4:
            val = 500.0
        elif pt == 5:
            val = 900.0
        else:
            val = 20000.0
        score += val if piece_is_white == ai_is_white else -val

    mob = 0
    for _ in board.legal_moves:
        mob += 1
    score += mob * 10 if board.turn == ai_is_white else -mob * 10

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
    cdef char hit
    cdef double val
    cdef object mover

    # 0) probe TT
    val = tt_probe(key, 0, alpha, beta, &hit)
    if hit:
        return val

    # 1) stand-pat
    if USE_NNUE:
        val = nnue_eval_py(acc.state)
    else:
        val = static_eval(board, acc, ai_color)

    # 2) alpha/beta check on stand-pat
    if val >= beta:
        tt_store(key, 0, val, LOWERBOUND)
        return beta
    if val > alpha:
        alpha = val

    # 3) only consider captures
    cdef object mv, captured
    cdef uint64_t next_key, cap_hash
    cdef double score
    cdef int from_pi, to_pi
    for mv in board.legal_moves:
        if not board.is_capture(mv):
            continue

        # who’s on from_square?
        mover = board.piece_at(mv.from_square)
        # who’s captured on to_square?
        captured = board.piece_at(mv.to_square)

        # incremental hash:
        #  a) remove mover from from_square
        from_pi = piece_index(mover.piece_type, mover.color)
        to_pi   = piece_index(mv.promotion if mv.promotion else mover.piece_type,
                                    mover.color)
        next_key = key ^ zobrist_random[from_pi * 64 + mv.from_square]
        # if there is a capture, remove the captured piece
        if captured is not None:
            cap_hash = zobrist_random[piece_index(captured.piece_type, captured.color) * 64 + mv.to_square]
        else:
            cap_hash = 0
        next_key ^= cap_hash
        # add mover at to_square (handle promotions by using to_pi)
        next_key ^= zobrist_random[to_pi * 64 + mv.to_square]
        # flip the side‐to‐move bit
        next_key ^= zobrist_random[768]
        

        # do the capture
        board.push(mv)
        acc.update(mv, captured)

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
    Negamax with alpha-beta, TT, and incremental Zobrist hashing.
    - `key` is the current zobrist hash for `board`.
    """
    global nodes_evaluated, branches_pruned
    cdef double value, child, cached
    cdef object mv, captured, mover
    cdef uint64_t next_key, cap_hash
    cdef char hit
    cdef int from_pi, to_pi

    # Count nodes
    nodes_evaluated += 1

    # 1) TT probe
    cached = tt_probe(key, required_depth, alpha, beta, &hit)
    if hit:
        return cached

    # 2) Terminal:
    if board.is_game_over():
        if board.is_checkmate():
            # if side‐to‐move is mated, return a mate score
            return -MATE_SCORE + depth
        else:
            return 0.0

    # 3) Leaf → quiescence
    if depth == 0:
        child = quiesce(board, acc, alpha, beta, ai_color, key)
        tt_store(key, 0, child, EXACT)
        return child

    # 4) Negamax loop
    value = -INFINITY
    for mv in board.legal_moves:
        # skip non-captures if you want deeper quiescence, etc.
        # (here we do full moves)
        mover    = board.piece_at(mv.from_square)
        captured = board.piece_at(mv.to_square)

        # incremental Zobrist:
        #   remove mover @ from, remove captured @ to, add mover @ to
        from_pi = piece_index(mover.piece_type, mover.color)
        to_pi   = piece_index(mv.promotion if mv.promotion else mover.piece_type,
                                    mover.color)
        next_key = key ^ zobrist_random[from_pi * 64 + mv.from_square]
        # if there is a capture, remove the captured piece
        if captured is not None:
            cap_hash = zobrist_random[piece_index(captured.piece_type, captured.color) * 64 + mv.to_square]
        else:
            cap_hash = 0
        next_key ^= cap_hash
        # add mover at to_square (handle promotions by using to_pi)
        next_key ^= zobrist_random[to_pi * 64 + mv.to_square]
        # flip the side‐to‐move bit
        next_key ^= zobrist_random[768]

        board.push(mv)
        acc.update(mv, captured)

        # negamax recursive call with swapped alpha/beta
        child = -minimax(board, acc,
                         depth-1,
                         -beta, -alpha,
                         ai_color,
                         next_key,
                         required_depth)

        board.pop()
        acc.rollback(mv, captured)

        if child > value:
            value = child
        if value > alpha:
            alpha = value

        if alpha >= beta:
            # cutoff
            branches_pruned += 1
            tt_store(key, depth, value, LOWERBOUND)
            return value

    # 5) store exact and return
    tt_store(key, depth, value, EXACT)
    return value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef SearchResult search_root(str fen,
                              int depth,
                              bint maximize,
                              str ai_color):
    # — declarations first —
    cdef object root, acc

    cdef double  bestv, v
    cdef object  bestmove
    cdef object  mv, mover, captured
    cdef int     from_pi, to_pi
    cdef uint64_t child_key

    # — now the code —
    root = Board(fen)
    root.turn = (ai_color == "white")

    acc = Accumulator()
    acc.init(root)

    root_key = <uint64_t> zobrist_hash(root)

    bestv    = -INFINITY if maximize else INFINITY
    bestmove = None

    for mv in root.legal_moves:
        # pull pieces out of root
        mover    = root.piece_at(mv.from_square)
        captured = root.piece_at(mv.to_square)

        # compute child_key exactly as in minimax/quiesce
        from_pi   = piece_index(mover.piece_type, mover.color)
        to_pi     = piece_index(mv.promotion if mv.promotion else mover.piece_type,
                                mover.color)
        child_key = (root_key
                     ^ zobrist_random[from_pi * 64 + mv.from_square]
                     ^ (captured and zobrist_random[
                            piece_index(captured.piece_type, captured.color) * 64
                            + mv.to_square])
                     ^ zobrist_random[to_pi   * 64 + mv.to_square]
                     ^ zobrist_random[768]  # flip side-to-move
                    )

        # make the move
        root.push(mv)
        acc.update(mv, captured)

        # call your negamax
        v = minimax(root, acc,
                    depth-1,
                    -INFINITY, INFINITY,
                    ai_color,
                    child_key,
                    depth)

        # undo
        root.pop()
        acc.rollback(mv, captured)

        # track best
        if (maximize and v > bestv) or ((not maximize) and v < bestv):
            bestv, bestmove = v, mv

    return SearchResult(bestv, bestmove)

# Allocate a 4M-entry table by default
init_tt(1<<22)