#!/usr/bin/env python3
"""
Simple profiler: Run a search on a position and profile everything.
"""

import cProfile
import pstats
import io
import sys
import os

# Setup paths
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
os.chdir(os.path.join(script_dir, 'src'))

def search_position():
    """Run a search on a position."""
    import chess
    from accumulator import Accumulator
    import core_search

    # Initialize
    core_search.set_use_nnue(True)
    core_search.init_nnue("nnue/halfkp_int8.pt")

    # Create board at a middlegame position
    board = chess.Board()
    for move_uci in ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'f1e2', 'e7e6']:
        board.push(chess.Move.from_uci(move_uci))

    acc = Accumulator()
    acc.init(board)

    print(f"\nSearching position: {board.fen()[:40]}...")
    print(f"Side to move: {['White', 'Black'][not board.turn]}\n")

    # Run search depth 6
    root_hash = core_search.compute_hash(board)
    ai_color = 'white' if board.turn == chess.WHITE else 'black'

    result = core_search.minimax(
        board, acc,
        depth=10,
        alpha=float('-inf'),
        beta=float('inf'),
        ai_color=ai_color,
        key=root_hash,
        required_depth=10
    )

    print(f"Result: {result:.4f}\n")

def profile_recompute_bottleneck():
    """Profile what's expensive in _recompute: FEN parsing or embedding summing."""
    import chess
    from accumulator import Accumulator
    from nnue.halfkp import halfkp_indices_for_fen
    import numpy as np
    import time

    print("\n" + "="*70)
    print("RECOMPUTE BOTTLENECK ANALYSIS")
    print("="*70)

    board = chess.Board()
    for move_uci in ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'f1e2', 'e7e6']:
        board.push(chess.Move.from_uci(move_uci))

    acc = Accumulator()
    acc.init(board)
    w = acc._load_weights() if hasattr(acc, '_load_weights') else None

    # Load weights at module level for testing
    from accumulator import _load_weights
    w = _load_weights()

    # Run 1000 recomputes and time each phase
    n_iters = 1000

    t0 = time.perf_counter()
    for _ in range(n_iters):
        raw0, raw1 = halfkp_indices_for_fen(board.fen())
    t_fen = time.perf_counter() - t0
    print(f"\n1. FEN parsing ({n_iters}x halfkp_indices_for_fen): {t_fen:.3f}s")
    print(f"   Per call: {t_fen/n_iters*1000:.2f}ms")

    # Simulate embedding summing (compute indices once, sum 1000x)
    raw0, raw1 = halfkp_indices_for_fen(board.fen())
    idx0 = acc._pad(raw0)
    idx1 = acc._pad(raw1)

    t0 = time.perf_counter()
    for _ in range(n_iters):
        h0 = w['emb0'][idx0].sum(axis=0).copy()
        h1 = w['emb1'][idx1].sum(axis=0).copy()
    t_embed = time.perf_counter() - t0
    print(f"\n2. Embedding summing ({n_iters}x): {t_embed:.3f}s")
    print(f"   Per call: {t_embed/n_iters*1000:.2f}ms")

    # Full _recompute (includes both)
    t0 = time.perf_counter()
    for _ in range(n_iters):
        acc._recompute()
    t_full = time.perf_counter() - t0
    print(f"\n3. Full _recompute ({n_iters}x): {t_full:.3f}s")
    print(f"   Per call: {t_full/n_iters*1000:.2f}ms")

    print(f"\n{'FEN parsing':25} {t_fen/(t_fen+t_embed)*100:.1f}% of bottleneck")
    print(f"{'Embedding summing':25} {t_embed/(t_fen+t_embed)*100:.1f}% of bottleneck")
    print(f"{'Overhead/other':25} {(t_full - t_fen - t_embed)/(t_fen+t_embed)*100:.1f}%")

if __name__ == '__main__':
    # First, profile the recompute bottleneck
    try:
        profile_recompute_bottleneck()
    except Exception as e:
        print(f"ERROR in recompute analysis: {e}")
        import traceback
        traceback.print_exc()

    # Then profile the full search
    pr = cProfile.Profile()
    pr.enable()

    try:
        search_position()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

    pr.disable()

    # Print results
    print("\n" + "="*70)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME")
    print("="*70)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

    print("\n" + "="*70)
    print("TOP 20 FUNCTIONS BY TOTAL TIME (where time actually spent)")
    print("="*70)
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(20)
    print(s.getvalue())

    # Save
    # pr.dump_stats('profile.prof')
    # print("Stats saved to profile.prof")
