import json

# Ruta del archivo original
input_file = "../data/cak.json"

# Ruta del archivo limpio de salida
output_file = "../data/cak_limpio.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Accedemos al array de entradas
entries = data.get("content", {}).get("body", [])

cleaned_entries = []

for entry in entries:
    examples = entry.get("examples", [])
    
    # Filtrar: solo entradas con ejemplos no vacíos
    if examples and len(examples) > 0:
        
        # Limpiar cada ejemplo eliminando "dialect"
        cleaned_examples = []
        for ex in examples:
            cleaned_example = {
                "example_maya": ex.get("example_maya"),
                "example_spa": ex.get("example_spa")
            }
            cleaned_examples.append(cleaned_example)

        cleaned_entry = {
            "entryWord": entry.get("entryWord"),
            "definition": entry.get("definition"),
            "examples": cleaned_examples
        }

        cleaned_entries.append(cleaned_entry)

# Construir nuevo JSON
cleaned_data = {
    "entries": cleaned_entries
}

# Guardar archivo limpio
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(cleaned_data, f, ensure_ascii=False, indent=2)

print(f"Archivo limpio guardado en: {output_file}")