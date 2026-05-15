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

# Build hierarchical tree structure
hierarchical_json = {
    "project": "Mini-Stockfish",
    "description": "Chess engine with NNUE neural network evaluation",
    "layers": []
}

# Group modules by layer
modules_by_layer = defaultdict(list)
for module in modules:
    modules_by_layer[module["layer"]].append(module)

# Map layer IDs to layer metadata
layer_map = {layer["id"]: layer for layer in layers}
component_map = {comp["id"]: comp for comp in components}

# Build hierarchical structure
for layer_id in ["ui", "game_logic", "search", "neural", "utilities"]:
    if layer_id not in layer_map:
        continue

    layer_data = layer_map[layer_id]
    layer_modules = modules_by_layer.get(layer_id, [])

    # Group modules by component
    modules_by_component = defaultdict(list)
    for module in layer_modules:
        modules_by_component[module["component"]].append(module)

    layer_entry = {
        "name": layer_data["name"],
        "id": layer_id,
        "description": layer_data["description"],
        "components": []
    }

    for comp_id, comp_modules in modules_by_component.items():
        if comp_id not in component_map:
            continue

        component_entry = {
            "name": component_map[comp_id]["name"],
            "id": comp_id,
            "modules": []
        }

        for module in comp_modules:
            module_entry = {
                "name": module["name"],
                "path": module["file"],
                "classes": module.get("classes", []),
                "dependencies": module.get("imports", []),
                "description": module.get("description", "")
            }
            component_entry["modules"].append(module_entry)

        layer_entry["components"].append(component_entry)

    # Handle modules from other layers (like core, data, etc) that have this layer
    other_layer_modules = [m for m in modules if m["layer"] not in layer_map]
    for module in other_layer_modules:
        if layer_id in module.get("layers", []):
            if not any(m["id"] == module["id"] for m in layer_modules):
                modules_by_layer[layer_id].append(module)

    hierarchical_json["layers"].append(layer_entry)

# Write both formats to the same file
output = {
    "nodeLink": node_link_json,
    "hierarchical": hierarchical_json
}

with open('architecture.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"Generated architecture.json with both node-link and hierarchical formats")
