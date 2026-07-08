import json
import unicodedata
import os
import re


# ===================== NORMALIZACIÓN BÁSICA =====================

def normalizar_unicode(texto):
    if not isinstance(texto, str):
        return texto

    # NFC
    texto = unicodedata.normalize("NFC", texto)

    # Unificar apóstrofes → ʼ (U+02BC)
    texto = texto.replace("'", "ʼ")

    return texto


def limpiar_texto_base(texto):
    texto = texto.strip()

    # Quitar "=" al inicio
    if texto.startswith("="):
        texto = texto[1:].strip()

    # Quitar paréntesis envolventes completos
    if texto.startswith("(") and texto.endswith(")"):
        texto = texto[1:-1].strip()

    # Normalizar espacios
    texto = re.sub(r"\s+", " ", texto)

    return texto


# ===================== EXTRACCIÓN DE VARIANTES =====================

def extraer_variantes(entry_word):
    """
    Devuelve:
    - canonical
    - variants (lista, puede ser vacía)
    """

    texto = normalizar_unicode(entry_word)
    texto = limpiar_texto_base(texto)

    variantes = []

    # -------- CASO 1: paréntesis --------
    parentesis = re.findall(r"\((.*?)\)", texto)

    for p in parentesis:
        p = limpiar_texto_base(normalizar_unicode(p))
        if p:
            variantes.append(p)

    # eliminamos contenido entre paréntesis del texto base
    texto_sin_parentesis = re.sub(r"\(.*?\)", "", texto).strip()

    # -------- CASO 2: comas --------
    if "," in texto_sin_parentesis:
        partes = [p.strip() for p in texto_sin_parentesis.split(",") if p.strip()]

        canonical = limpiar_texto_base(partes[0])
        variantes.extend([limpiar_texto_base(p) for p in partes[1:]])

    else:
        # caso simple: no coma → todo es canonical
        canonical = limpiar_texto_base(texto_sin_parentesis)

    # limpiar duplicados y vacíos
    variantes = list(dict.fromkeys([v for v in variantes if v and v != canonical]))

    return canonical, variantes


# ===================== NORMALIZACIÓN DE ENTRIES =====================

def normalizar_entry(entry):
    nuevo_entry = dict(entry)

    # ---- ENTRY WORD ----
    entry_word = entry.get("entryWord", "")
    canonical, variantes = extraer_variantes(entry_word)

    nuevo_entry["entryWord"] = canonical

    if variantes:
        nuevo_entry["variants"] = variantes

    # ---- DEFINICIÓN ----
    if "definition" in nuevo_entry:
        nuevo_entry["definition"] = normalizar_unicode(
            limpiar_texto_base(nuevo_entry["definition"])
        )

    # ---- EJEMPLOS ----
    if "examples" in nuevo_entry:
        nuevos_ejemplos = []
        for ex in nuevo_entry["examples"]:
            nuevo_ex = dict(ex)

            if "example_maya" in nuevo_ex:
                nuevo_ex["example_maya"] = normalizar_unicode(
                    limpiar_texto_base(nuevo_ex["example_maya"])
                )

            if "example_spa" in nuevo_ex:
                nuevo_ex["example_spa"] = normalizar_unicode(
                    limpiar_texto_base(nuevo_ex["example_spa"])
                )

            nuevos_ejemplos.append(nuevo_ex)

        nuevo_entry["examples"] = nuevos_ejemplos

    return nuevo_entry


# ===================== PIPELINE PRINCIPAL =====================

def normalizar_json(ruta_entrada):
    with open(ruta_entrada, "r", encoding="utf-8") as f:
        data = json.load(f)

    nuevas_entries = []

    for entry in data.get("entries", []):
        nuevo_entry = normalizar_entry(entry)
        nuevas_entries.append(nuevo_entry)

    data["entries"] = nuevas_entries

    # ---- nombre salida ----
    base, ext = os.path.splitext(ruta_entrada)
    ruta_salida = f"{base}_normalizado{ext}"

    with open(ruta_salida, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Archivo normalizado guardado en: {ruta_salida}")


# ===================== MAIN =====================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Uso: python normalizar.py archivo.json")
    else:
        ruta = sys.argv[1]
        normalizar_json(ruta)