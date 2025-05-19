# distutils: language = c++
# distutils: libraries = nnue_inference
# distutils: library_dirs = nnue/build
# distutils: include_dirs = nnue

import os

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

# Python‐level counters
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
cdef double quiesce(object board, object acc,
                    double alpha, double beta,
                    str ai_color):
    # leaf: run the NNUE network on acc.state
    if USE_NNUE:
        return nnue_eval_py(acc.state)
    else:
        return static_eval(board, acc, ai_color)

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double minimax(object board, object acc,
                     int depth,
                     double alpha, double beta,
                     bint maximizing,
                     str ai_color):
    global nodes_evaluated, branches_pruned
    cdef double value, child
    cdef object mv, captured

    nodes_evaluated += 1
    if board.is_game_over():
        if board.is_checkmate():
            if maximizing:
                return -MATE_SCORE + depth
            # otherwise it's your opponent to move who is mated → best win
            else:
                return  MATE_SCORE - depth
        else:
            return 0.0

    if depth == 0:
        return quiesce(board, acc, alpha, beta, ai_color)

    value = -INFINITY if maximizing else INFINITY

    for mv in board.legal_moves:
        captured = board.piece_at(mv.to_square)
        board.push(mv)
        acc.update(mv, captured)

        child = minimax(board, acc,
                        depth-1,
                        alpha, beta,
                        not maximizing,
                        ai_color)

        acc.rollback(mv, captured)
        board.pop()

        if maximizing:
            if child > value:
                value = child
            if value > alpha:
                alpha = value
        else:
            if child < value:
                value = child
            if value < beta:
                beta = value

        if alpha >= beta:
            branches_pruned += 1
            break

    return value

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef SearchResult search_root(str fen, int depth, bint maximize, str ai_color):
    """
    Top‐level search entrypoint exposed to Python.
    """
    cdef object root = Board(fen)
    root.turn = True if ai_color == "white" else False
    cdef object acc = Accumulator()
    acc.init(root)

    cdef double bestv = -INFINITY if maximize else INFINITY
    cdef object bestmove = None, uci, captured, v

    for uci in root.legal_moves:
        captured = root.piece_at(uci.to_square)
        root.push(uci)
        acc.update(uci, captured)

        v = minimax(root, acc,
                    depth-1,
                    -INFINITY, INFINITY,
                    not maximize,
                    ai_color)

        acc.rollback(uci, captured)
        root.pop()

        if (maximize and v > bestv) or ((not maximize) and v < bestv):
            bestv, bestmove = v, uci

    return SearchResult(bestv, bestmove)
