#!/usr/bin/env python3
"""Test single depth-10 search (no iterative deepening)."""

import sys
import os
import time

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, 'src'))
os.chdir(os.path.join(script_dir, 'src'))

import chess
from accumulator import Accumulator
import core_search

# Initialize
core_search.set_use_nnue(True)
core_search.init_nnue("nnue/halfkp_int8.pt")
core_search.clear_tt()
core_search.reset_tt_counters()

# Create board
board = chess.Board()
for move_uci in ['e2e4', 'c7c5', 'g1f3', 'd7d6', 'f1e2', 'e7e6']:
    board.push(chess.Move.from_uci(move_uci))

acc = Accumulator()
acc.init(board)

print(f"Single depth-10 search (no iterative deepening)")
print(f"Position: {board.fen()[:40]}...\n")

root_hash = core_search.compute_hash(board)
ai_color = 'white'

t0 = time.perf_counter()
result = core_search.minimax(
    board, acc,
    depth=10,
    alpha=float('-inf'),
    beta=float('inf'),
    ai_color=ai_color,
    key=root_hash,
    required_depth=10
)
elapsed = time.perf_counter() - t0

print(f"Result: {result:.4f}")
print(f"Time: {elapsed:.2f}s")
print(f"Nodes evaluated: {core_search.get_nodes_evaluated():,}")
print(f"Branches pruned: {core_search.get_branches_pruned():,}")

tt_h = core_search.get_tt_hits()
tt_m = core_search.get_tt_misses()
tt_total = tt_h + tt_m
tt_pct = (100.0 * tt_h / tt_total) if tt_total > 0 else 0.0
print(f"TT hits: {tt_h:,}, misses: {tt_m:,}, hit rate: {tt_pct:.1f}%")
