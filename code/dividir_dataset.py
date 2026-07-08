import json
import random
import os

# =========================
# CONFIGURACIÓN
# =========================

INPUT_FILE = "../diccionarios/cak_limpio.json"

OUTPUT_TRAIN = "entrenamiento.json"
OUTPUT_STATIC = "entrenamiento_estatico.json"

OUTPUT_EVAL_1 = "evaluacion_1.json"
OUTPUT_EVAL_2 = "evaluacion_2.json"

STATIC_SIZE = 30
EVAL_SIZE = 250

RANDOM_SEED = 42

# =========================
# FUNCIONES
# =========================

def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "entries" not in data:
        raise ValueError("El JSON no contiene la clave 'entries'")

    return data["entries"]


def save_json(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"entries": entries}, f, ensure_ascii=False, indent=2)


def count_examples(entries):
    return sum(len(entry.get("examples", [])) for entry in entries)


def build_eval_set(entries, target_examples):
    selected_entries = []
    current_examples = 0
    used_ids = set()

    for entry in entries:
        entry_examples = len(entry.get("examples", []))

        if current_examples + entry_examples <= target_examples:
            selected_entries.append(entry)
            current_examples += entry_examples
            used_ids.add(id(entry))

        if current_examples == target_examples:
            break

    return selected_entries, used_ids


# =========================
# MAIN
# =========================

def main():

    entries = load_data(INPUT_FILE)

    print(f"Total de entradas: {len(entries)}")

    random.seed(RANDOM_SEED)

    random.shuffle(entries)

    # ==========================================
    # STATIC
    # ==========================================

    static_entries = entries[:STATIC_SIZE]

    remaining_entries = entries[STATIC_SIZE:]

    # ==========================================
    # EVAL 1
    # ==========================================

    eval_1_entries, eval_1_ids = build_eval_set(
        remaining_entries,
        EVAL_SIZE
    )

    remaining_after_eval_1 = [
        e for e in remaining_entries
        if id(e) not in eval_1_ids
    ]

    # ==========================================
    # EVAL 2
    # ==========================================

    eval_2_entries, eval_2_ids = build_eval_set(
        remaining_after_eval_1,
        EVAL_SIZE
    )

    # ==========================================
    # TRAIN
    # ==========================================

    train_entries = [
        e for e in remaining_after_eval_1
        if id(e) not in eval_2_ids
    ]

    # ==========================================
    # VERIFICACIONES
    # ==========================================

    datasets = {
        "train": train_entries,
        "static": static_entries,
        "eval1": eval_1_entries,
        "eval2": eval_2_entries
    }

    dataset_ids = {
        name: set(id(e) for e in data)
        for name, data in datasets.items()
    }

    names = list(dataset_ids.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert dataset_ids[names[i]].isdisjoint(dataset_ids[names[j]])

    # ==========================================
    # GUARDAR
    # ==========================================

    save_json(OUTPUT_TRAIN, train_entries)
    save_json(OUTPUT_STATIC, static_entries)
    save_json(OUTPUT_EVAL_1, eval_1_entries)
    save_json(OUTPUT_EVAL_2, eval_2_entries)

    # ==========================================
    # RESUMEN
    # ==========================================

    print("\nProceso completado correctamente:")

    for name, data in datasets.items():
        print(f"\n{name}")
        print(f"   - EntryWords: {len(data)}")
        print(f"   - Examples: {count_examples(data)}")


# =========================
# EJECUCIÓN
# =========================

if __name__ == "__main__":
    main()