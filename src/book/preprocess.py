import glob
import json
import io
import time

import chess
import chess.pgn
from stockfish import Stockfish
from chess.polyglot import zobrist_hash


# 1) configure your Stockfish binary & desired search depth
sf = Stockfish(
    path="C:/Users/daris/Downloads/stockfish/stockfish-windows-x86-64-avx2.exe",
    depth=22,  # you can bump this back up once you verify it works
)

eval_map   = {}        # zobrist_key (int) → eval (int, centipawns)
seen_keys  = set()     # to avoid duplicate analyses
total_seen = 0

start = time.time()
for tsv_path in glob.glob("src/book/*.tsv"):
    print(f"\n→ Processing file: {tsv_path}")
    with open(tsv_path, "r") as tsv:
        next(tsv)  # skip header row
        for line_no, line in enumerate(tsv, start=2):
            eco, name, pgn_moves = line.strip().split("\t")

            # debug: show current file and line number
            print(f"[Line {line_no}] {eco} – {name}", end="\r")

            # build a tiny PGN so python-chess can parse it
            pgn = io.StringIO(f"[Event \"{eco}\"]\n\n1. {pgn_moves} *")
            game = chess.pgn.read_game(pgn)
            board = game.board()

            # walk the mainline up to 10 plies
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= 10:
                    break

                total_seen += 1
                key = zobrist_hash(board)

                if key not in seen_keys:
                    # update the same line-within-file print with counter
                    print(
                        f"[{total_seen:5d}] File {tsv_path!r}, "
                        f"Line {line_no}, Ply {ply+1}",
                        end="\r"
                    )

                    sf.set_fen_position(board.fen())
                    cp = sf.get_evaluation()["value"]

                    eval_map[key]   = cp
                    seen_keys.add(key)

                board.push(move)

elapsed = time.time() - start
print(
    f"\nDone: processed {total_seen} plies, "
    f"{len(eval_map)} unique positions in {elapsed:.1f}s"
)

# 4) serialize to JSON
with open("src/book/book.json", "w") as out:
    json.dump({str(k): v for k, v in eval_map.items()}, out)

print(f"Dumped {len(eval_map)} positions to book.json")
