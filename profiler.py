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
    """Run iterative deepening search on a position (like ai.py)."""
    import chess
    from accumulator import Accumulator
    import core_search
    import time
    import math

    # Initialize
    core_search.set_use_nnue(True)
    core_search.init_nnue("nnue/halfkp_int8.pt")

    # Create board at a middlegame position
    board = chess.Board()
    for move_uci in ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'f1e2', 'e7e6']:
        board.push(chess.Move.from_uci(move_uci))

    root_fen = board.fen()
    ai_color = 'white' if board.turn == chess.WHITE else 'black'

    print(f"\nSearching position: {root_fen[:40]}...")
    print(f"Side to move: {['White', 'Black'][not board.turn]}\n")

    # Clear TT and start iterative deepening
    core_search.reset_tt_counters()
    core_search.clear_tt()
    core_search.clear_history()

    t0 = time.perf_counter()

    # Iterative deepening: depths 1 through 11 (to reach depth=10 search)
    # Match ai.py logic: for depth in range(1, self.depth + 1), call minimax(..., depth - 1, ...)
    search_depth = 11
    best_move = None
    best_eval = -math.inf

    last_finite_eval = None
    for depth in range(1, search_depth + 1):
        core_search.reset_counters()
        core_search.clear_killers()
        # Keep TT between depths — that's the whole point of iterative deepening

        root_board = chess.Board(root_fen)
        root_board.turn = chess.WHITE if ai_color == 'white' else chess.BLACK
        moves = list(root_board.legal_moves)

        # ——— Aspiration Windows ———
        asp_delta = 0.5
        if (depth >= 3
            and last_finite_eval is not None
            and not math.isinf(last_finite_eval)
            and abs(last_finite_eval) < 100000):
            asp_alpha = last_finite_eval - asp_delta
            asp_beta = last_finite_eval + asp_delta
        else:
            asp_alpha = -math.inf
            asp_beta = math.inf

        asp_attempts = 0
        while True:
            alpha = asp_alpha
            failed_high = False

            for uci in moves:
                root_board.push(uci)
                acc = Accumulator()
                acc.init(root_board)

                root_key = core_search.compute_hash(root_board)

                # Negamax: negate child score, use aspiration window
                val = -core_search.minimax(
                    root_board, acc,
                    depth - 1,
                    -asp_beta, -alpha,
                    ai_color,
                    root_key,
                    depth
                )

                root_board.pop()

                if val > alpha:
                    alpha = val
                    best_move = uci

                if alpha >= asp_beta:
                    failed_high = True
                    break

            asp_attempts += 1
            if not math.isinf(asp_alpha) and alpha <= asp_alpha:
                asp_delta *= 4
                if asp_attempts >= 2:
                    asp_alpha = -math.inf
                else:
                    asp_alpha = last_finite_eval - asp_delta
                continue
            elif failed_high or (not math.isinf(asp_beta) and alpha >= asp_beta):
                asp_delta *= 4
                if asp_attempts >= 2:
                    asp_beta = math.inf
                else:
                    asp_beta = last_finite_eval + asp_delta
                continue
            else:
                break

        best_eval = alpha
        if not math.isinf(best_eval) and abs(best_eval) < 100000:
            last_finite_eval = best_eval

        # Print per-depth stats
        tt_h = core_search.get_tt_hits()
        tt_m = core_search.get_tt_misses()
        tt_total = tt_h + tt_m
        tt_pct = (100.0 * tt_h / tt_total) if tt_total > 0 else 0.0
        nodes = core_search.get_nodes_evaluated()
        pruned = core_search.get_branches_pruned()

        print(f"Depth {depth:2d}: eval={best_eval:7.4f}  nodes={nodes:8,}  pruned={pruned:7,}  TT: {tt_pct:5.1f}%")

    elapsed = time.perf_counter() - t0

    print(f"\nTotal time: {elapsed:.3f}s")

    # Final TT stats
    tt_hits = core_search.get_tt_hits()
    tt_misses = core_search.get_tt_misses()
    total_probes = tt_hits + tt_misses
    tt_hitrate = (tt_hits / total_probes * 100) if total_probes > 0 else 0

    print(f"Final eval: {best_eval:.4f}")
    print(f"TT probes: {total_probes:,} (hits: {tt_hits:,}, misses: {tt_misses:,})")
    print(f"TT hit rate: {tt_hitrate:.1f}%")

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
