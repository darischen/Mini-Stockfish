import json
from collections import defaultdict

# Load extracted metadata
with open('architecture-data.json', 'r') as f:
    data = json.load(f)

modules = data['modules']

# Define layers with metadata
layers = [
    {"id": "ui", "name": "UI Layer", "color": "#ADD8E6", "description": "User interface and game interaction"},
    {"id": "game_logic", "name": "Game Logic Layer", "color": "#90EE90", "description": "Chess rules and board representation"},
    {"id": "search", "name": "Search Engine Layer", "color": "#FFA500", "description": "Minimax search with alpha-beta pruning"},
    {"id": "neural", "name": "Neural Network Layer", "color": "#DDA0DD", "description": "NNUE evaluation and training"},
    {"id": "utilities", "name": "Utilities & Preprocessing", "color": "#D3D3D3", "description": "Helper modules and data processing"},
]

# Define components
components = [
    {"id": "game", "name": "Game Components", "layers": ["ui", "game_logic"], "description": "Game state, board, pieces, and user interaction"},
    {"id": "ai_engine", "name": "AI Engine", "layers": ["search"], "description": "Minimax search, move ordering, and pruning"},
    {"id": "neural_network", "name": "Neural Network", "layers": ["neural"], "description": "NNUE training and evaluation"},
    {"id": "utilities", "name": "Utilities", "layers": ["utilities"], "description": "Configuration, preprocessing, and helper modules"},
]

# Build nodes from modules
nodes = []
for module in modules:
    node = {
        "id": module["id"],
        "name": module["name"],
        "file": module["file"],
        "layer": module["layer"],
        "component": module["component"],
        "classes": module.get("classes", []),
        "description": module.get("description", ""),
        "imports_count": len(module.get("imports", [])),
        "imported_by_count": len(module.get("imported_by", [])),
    }
    nodes.append(node)

# Build edges from dependencies
edges = []
for module in modules:
    for imported in module.get("imports", []):
        edges.append({
            "source": module["id"],
            "target": imported,
            "type": "import",
            "description": f"{module['id']} imports {imported}"
        })

# Create node-link JSON
node_link_json = {
    "metadata": {
        "project": "Mini-Stockfish",
        "version": "1.0",
        "generated": "2026-05-15",
        "description": "Chess engine with NNUE neural network evaluation"
    },
    "layers": layers,
    "components": components,
    "nodes": nodes,
    "edges": edges
}

# Write node-link format
with open('architecture.json', 'w') as f:
    json.dump(node_link_json, f, indent=2)

print(f"Generated architecture.json with {len(nodes)} nodes and {len(edges)} edges")
