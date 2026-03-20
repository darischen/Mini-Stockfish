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

if __name__ == '__main__':
    # Profile the search
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
    print("="*70)
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
