import json
import chess
import chess.pgn
from chess.polyglot import zobrist_hash

# Path to your PGN file
PGN_PATH = "endgames.pgn"
# Output JSON file
JSON_PATH = "tablebase.json"

def parse_moves(tag_value):
    """Split the comma-separated moves or return an empty list."""
    if not tag_value:
        return []
    return [move.strip() for move in tag_value.split(",") if move.strip()]

def build_endgame_book(pgn_path):
    endgame_book = {}
    with open(pgn_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            fen = game.headers.get("FEN")
            if not fen:
                continue
            board = chess.Board(fen)
            key = zobrist_hash(board)
            wdl = game.headers.get("WDL")
            dtz = int(game.headers.get("DTZ", "0"))
            winning = parse_moves(game.headers.get("WinningMoves", ""))
            drawing = parse_moves(game.headers.get("DrawingMoves", ""))
            losing = parse_moves(game.headers.get("LosingMoves", ""))
            
            endgame_book[str(key)] = {
                "wdl": wdl,
                "dtz": dtz,
                "winning": winning,
                "drawing": drawing,
                "losing": losing
            }
    return endgame_book

# Build the book and write out to JSON
book = build_endgame_book(PGN_PATH)
with open(JSON_PATH, 'w') as json_file:
    json.dump(book, json_file, indent=2)

print(f"Endgame book written to {JSON_PATH}")
