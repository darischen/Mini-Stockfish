import json
from collections import defaultdict

# Load extracted metadata
with open('architecture-data.json', 'r') as f:
    data = json.load(f)

modules = data['modules']

# Define layers with metadata
layers = [
    {"id": "ui", "name": "UI Layer", "color": "#ADD8E6", "description": "User interface and game interaction"},
    {"id": "core", "name": "Core Game Logic Layer", "color": "#90EE90", "description": "Chess rules and board representation"},
    {"id": "search", "name": "Search Engine Layer", "color": "#FFA500", "description": "Minimax search with alpha-beta pruning"},
    {"id": "nnue", "name": "Neural Network Layer", "color": "#DDA0DD", "description": "NNUE evaluation and training"},
    {"id": "data", "name": "Data Processing Layer", "color": "#D3D3D3", "description": "Data preprocessing and helper modules"},
]

# Build unique components from modules
component_ids = set(module["component"] for module in modules)
components = []
for comp_id in sorted(component_ids):
    # Find the first module with this component to get description
    comp_modules = [m for m in modules if m["component"] == comp_id]
    comp_layer = comp_modules[0]["layer"] if comp_modules else "unknown"
    components.append({
        "id": comp_id,
        "name": " ".join(word.capitalize() for word in comp_id.replace("_", " ").split()),
        "layers": [comp_layer],
        "description": f"{comp_id} component modules"
    })

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

# Get all unique layer IDs from modules and sort them
actual_layer_ids = sorted(set(module["layer"] for module in modules))

# Build hierarchical structure
for layer_id in actual_layer_ids:
    layer_data = layer_map.get(layer_id)
    if not layer_data:
        # Skip layers not in the predefined layer list
        continue

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
