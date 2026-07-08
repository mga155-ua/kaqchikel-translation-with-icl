import json
import random
import sys
import math
import bm25s
import time
import os
import re
from google import genai
from google.genai import types

# ===================== CONFIG =====================

NUM_EJEMPLOS_TRAIN = 30

RUTA_ENTRENAMIENTO = "../data/entrenamiento_normalizado.json"
RUTA_ENTRENAMIENTO_ESTATICO = "../data/entrenamiento_estatico_normalizado.json"
RUTA_EVALUACION = "../data/evaluacion_1_normalizado.json"
RUTA_APIKEY = "../apikey.txt"
MAX_PETICIONES_API = 1400

RUTA_BASE_RESULTADOS = "../resultados/resultados_separados/frases_individuales1"
RUTA_LOGS = "../resultados/logs"
RUTA_ERRORES = f"{RUTA_LOGS}/errores.txt"

MODEL_NAME = "gemma-4-31b-it"

PARAMETROS_VALIDOS = {
    "def-propia",
    "def-cache",
    "ejemplo-cache",
    "def-ed",
    "ejemplo-ed",
    "def-bm",
    "ejemplo-bm",
}

EXPERIMENTOS = {
    frozenset(): "experimento1",
    frozenset(["def-propia"]): "experimento2",
    frozenset(["def-cache"]): "experimento3",
    frozenset(["def-propia", "def-cache"]): "experimento4",
    frozenset(["ejemplo-cache"]): "experimento5",
    frozenset(["ejemplo-cache", "def-propia"]): "experimento6",
    frozenset(["def-cache", "ejemplo-cache"]): "experimento7",
    frozenset(["def-ed"]): "experimento8",
    frozenset(["ejemplo-ed"]): "experimento9",
    frozenset(["def-ed", "ejemplo-ed"]): "experimento10",
    frozenset(["def-cache", "def-ed"]): "experimento11",
    frozenset(["ejemplo-cache", "ejemplo-ed"]): "experimento12",
    frozenset(["def-cache", "def-ed", "ejemplo-cache", "ejemplo-ed"]): "experimento13",
    frozenset(["def-bm"]): "experimento14",
    frozenset(["ejemplo-bm"]): "experimento15",
    frozenset(["def-bm", "ejemplo-bm"]): "experimento16",
    frozenset([
        "def-propia", "def-ed", "ejemplo-ed", "def-bm", "ejemplo-bm"
    ]): "experimento17",
    frozenset(["def-ed", "ejemplo-bm"]): "experimento18",
    frozenset(["def-bm", "ejemplo-ed"]): "experimento19",
    frozenset(["def-propia", "def-ed"]): "experimento20",
    frozenset(["def-propia", "ejemplo-bm"]): "experimento21",
    frozenset(["def-propia", "def-ed", "ejemplo-bm"]): "experimento22",
}
# ===================== CONTEXTO =====================

CONTEXTOS_POR_FRASE = []
CACHE_DEFINICIONES = []
CACHE_EJEMPLOS = []

# ===================== UTILIDADES =====================

def samplear(lista, k):
    if len(lista) <= k:
        return lista
    return random.sample(lista, k)

def tokenizar(texto):
    return texto.strip().lower().split()

def normalizar(texto):
    return texto.lower().replace("ʼ", "'").replace("‘", "'").replace("`", "'").strip()

def obtener_nombre_experimento(parametros):
    return EXPERIMENTOS.get(frozenset(parametros), "experimento_desconocido")

def separar_parametros_thinking(parametros):
    thinking = "thinking" in parametros
    parametros_filtrados = [p for p in parametros if p != "thinking"]
    return thinking, parametros_filtrados


# ===================== BM25 =====================

def inicializar_bm25(ejemplos_train):
    global BM25_RETRIEVER, BM25_CORPUS_MAYA, BM25_CORPUS_META

    corpus_maya = []
    corpus_meta = []

    for maya, spa, palabra, definicion in ejemplos_train:
        corpus_maya.append(maya)
        corpus_meta.append((spa, palabra, definicion))

    tokens = bm25s.tokenize(corpus_maya)
    retriever = bm25s.BM25()
    retriever.index(tokens)

    BM25_RETRIEVER = retriever
    BM25_CORPUS_MAYA = corpus_maya
    BM25_CORPUS_META = corpus_meta


def obtener_mejores_por_bm(frase_query, k=3):
    q_tokens = bm25s.tokenize([frase_query])
    results, scores = BM25_RETRIEVER.retrieve(q_tokens, k=k)

    results = results[0]
    scores = scores[0]

    output = []
    # print(f"\n[DEBUG][BM25] Query (frase completa): {frase_query}")

    for doc_id, score in zip(results, scores):
        maya = BM25_CORPUS_MAYA[doc_id]
        spa, palabra, definicion = BM25_CORPUS_META[doc_id]
        # print(f"[DEBUG][BM25] Rank | score={round(score,4)}")
        # print(f"[DEBUG][BM25]   Maya: {maya}")
        # print(f"[DEBUG][BM25]   Spa : {spa}")
        # print(f"[DEBUG][BM25]   Word: {palabra}")
        # print(f"[DEBUG][BM25]   Def : {definicion}")
        output.append((maya, spa, palabra, definicion, score))

    return output

# ===================== EDIT DISTANCE =====================

def distancia_edicion_normalizada(s1, s2):

    len1, len2 = len(s1), len(s2)

    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i

    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):

            coste = (
                0
                if s1[i - 1] == s2[j - 1]
                else 1
            )

            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + coste
            )

    distancia = dp[len1][len2]

    longitud_max = max(len1, len2)

    if longitud_max == 0:
        return 0.0

    return distancia / longitud_max


def obtener_relevantes_por_ed(
    frase,
    definiciones_train,
    top_k=4
):

    tokens = []

    for t in tokenizar(frase):

        t_norm = normalizar(t)

        if len(t_norm) < 3:
            continue

        tokens.append(t_norm)

    candidatos_globales = []

    # print("\n===================================")
    # print("[DEBUG][ED] NUEVA FRASE")
    # print("===================================")

    # print(f"[DEBUG][ED] Frase original:")
    # print(frase)

    # print(f"\n[DEBUG][ED] Tokens:")
    # print(tokens)

    for palabra, definicion, variantes in definiciones_train:

        candidatos = (
            [normalizar(palabra)]
            + [
                normalizar(v)
                for v in variantes
            ]
        )

        mejor_dist = math.inf
        mejor_token = None
        mejor_candidato = None

        for token in tokens:

            for cand in candidatos:

                dist = distancia_edicion_normalizada(
                    token,
                    cand
                )

                if dist < mejor_dist:

                    mejor_dist = dist
                    mejor_token = token
                    mejor_candidato = cand

        candidatos_globales.append({
            "palabra": palabra,
            "definicion": definicion,
            "distancia": mejor_dist,
            "token": mejor_token,
            "candidato": mejor_candidato
        })

    candidatos_globales.sort(
        key=lambda x: x["distancia"]
    )

    top_resultados = candidatos_globales[:top_k]

    # print("\n[DEBUG][ED] TOP RESULTADOS")

    for i, r in enumerate(top_resultados, 1):

        similitud = 1 - r["distancia"]

        # print(
        #     f"\n[{i}] "
        #     f"token='{r['token']}' "
        #     f"vs "
        #     f"candidato='{r['candidato']}'"
        # )

        # print(
        #     f"dist_normalizada="
        #     f"{r['distancia']:.4f}"
        # )

        # print(
        #     f"similitud="
        #     f"{similitud:.4f}"
        # )

        # print(
        #     f"palabra={r['palabra']}"
        # )

        # print(
        #     f"definicion={r['definicion']}"
        # )

    return [
        (
            r["palabra"],
            r["definicion"]
        )
        for r in top_resultados
    ]

# ===================== DATOS =====================

def cargar_datos(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        data = json.load(f)

    definiciones = []
    ejemplos = []

    for entry in data.get("entries", []):
        palabra = entry.get("entryWord")
        definicion = entry.get("definition")
        variantes = entry.get("variants", [])

        if palabra and definicion:
            definiciones.append((palabra.strip(), definicion.strip(), [v.strip() for v in variantes]))

        for ex in entry.get("examples", []):
            maya = ex.get("example_maya")
            spa = ex.get("example_spa")

            if maya and spa and palabra and definicion:
                ejemplos.append((maya.strip(), spa.strip(), palabra.strip(), definicion.strip()))

    return definiciones, ejemplos

def contar_traducciones_realizadas(ruta_resultado):

    if not os.path.exists(ruta_resultado):
        return 0

    contador = 0

    with open(ruta_resultado, "r", encoding="utf-8") as f:

        for linea in f:

            if linea.startswith("PRED:"):
                contador += 1

    return contador
# ===================== CONFIG EXPERIMENTO =====================

def configurar_experimento(parametros, definiciones_train, ejemplos_train, ejemplos_eval):
    global CONTEXTOS_POR_FRASE, CACHE_DEFINICIONES, CACHE_EJEMPLOS

    CONTEXTOS_POR_FRASE = []
    CACHE_DEFINICIONES = []
    CACHE_EJEMPLOS = []

    if "def-cache" in parametros:
        definiciones_cache, _ = cargar_datos(RUTA_ENTRENAMIENTO_ESTATICO)
        CACHE_DEFINICIONES = samplear(definiciones_cache, NUM_EJEMPLOS_TRAIN)

    if "ejemplo-cache" in parametros:
        _, ejemplos_cache = cargar_datos(RUTA_ENTRENAMIENTO_ESTATICO)
        CACHE_EJEMPLOS = samplear(ejemplos_cache, NUM_EJEMPLOS_TRAIN)

    for maya, spa, palabra_gt, definicion_gt in ejemplos_eval:

        defs = []
        ejems = []

        if "def-propia" in parametros:
            defs.append((palabra_gt, definicion_gt))

        if "def-ed" in parametros or "ejemplo-ed" in parametros:
            relevantes = obtener_relevantes_por_ed(maya, definiciones_train)

            if "def-ed" in parametros:
                defs.extend(relevantes)

            if "ejemplo-ed" in parametros:
                palabras = {p for p, _ in relevantes}
                for m, s, p, d in ejemplos_train:
                    if p in palabras:
                        ejems.append((m, s))

        if "def-bm" in parametros or "ejemplo-bm" in parametros:
            mejores = obtener_mejores_por_bm(maya)

            for m, s, p, d, _ in mejores:
                if "def-bm" in parametros:
                    defs.append((p, d))
                if "ejemplo-bm" in parametros:
                    ejems.append((m, s))

        CONTEXTOS_POR_FRASE.append({
            "frase": maya,
            "defs": defs[:5],
            "ejems": ejems[:5]
        })

# ===================== PROMPT =====================

def crear_prompt(ctx):

    bloque_cache = ""

    if CACHE_DEFINICIONES or CACHE_EJEMPLOS:
        bloque_cache += "\n========================\nCONTEXTO GLOBAL\n========================\n"

        if CACHE_DEFINICIONES:
            bloque_cache += "\n--- DEFINICIONES ---\n"
            for palabra, definicion, *_ in CACHE_DEFINICIONES:
                bloque_cache += f"{palabra} -> {definicion}\n"

        if CACHE_EJEMPLOS:
            bloque_cache += "\n--- EJEMPLOS ---\n"
            for maya, spa, *_ in CACHE_EJEMPLOS:
                bloque_cache += f"{maya} -> {spa}\n"

    bloque_frase = f"""
========================
FRASE
========================

Texto: {ctx['frase']}

"""

    if ctx["defs"] or ctx["ejems"]:
        bloque_frase += "Contexto relevante:\n"

        for palabra, definicion in ctx["defs"]:
            bloque_frase += f"- {palabra} -> {definicion}\n"

        for maya, spa in ctx["ejems"]:
            bloque_frase += f"- Ejemplo: {maya} -> {spa}\n"

    return f"""
Eres un traductor experto de Kaqchikel a español.

========================
FORMATO DE RESPUESTA
========================
Devuelve ÚNICAMENTE la traducción.

NO añadas explicaciones.

========================
EJEMPLO
========================

Entrada:
Esto sería una frase en kaqchikel

Salida:
Esto sería esa misma frase traducida al español

{bloque_cache}

========================
TAREA
========================

Traduce la siguiente frase:

{bloque_frase}

========================
RESPUESTA
========================
"""

# ===================== LLM =====================

def traducir(
    client,
    prompt,
    contador_peticiones,
    nombre_experimento,
    frase_maya,
    thinking=False,
    max_reintentos=10
):

    intento = 0

    while intento < max_reintentos:

        try:

            if contador_peticiones >= MAX_PETICIONES_API:

                raise Exception(
                    "Límite diario de peticiones alcanzado"
                )

            print(f"[INFO] Llamada API (intento {intento + 1}/{max_reintentos})")

            contador_peticiones += 1
            if thinking:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(
                            thinking_level="high"
                        )
                    ),
                )
            else:
                response = client.models.generate_content(
                    model=MODEL_NAME,
                    contents=prompt,
                )

            return response.text.strip(), contador_peticiones

        except Exception as e:

            intento += 1

            print(f"[WARNING] Error API:")
            print(f"[WARNING] {e}")

            if intento >= max_reintentos:

                print("[ERROR] Máximo número de reintentos alcanzado")

                guardar_error(
                    RUTA_ERRORES,
                    nombre_experimento,
                    frase_maya,
                    e
                )

                raise

            espera = min(300, (2 ** intento) + random.uniform(0, 3))

            print(f"[INFO] Reintentando en {round(espera, 2)} segundos...")
            time.sleep(espera)

def recuperar_errores(
    client,
    definiciones_train,
    ejemplos_train,
    ejemplos_eval
):

    print("\n[INFO] MODO RECUPERACIÓN\n")

    # =====================
    # LIMPIAR DUPLICADOS
    # =====================

    for nombre in os.listdir(
        RUTA_BASE_RESULTADOS
    ):

        if not nombre.endswith(".txt"):
            continue

        ruta = os.path.join(
            RUTA_BASE_RESULTADOS,
            nombre
        )

        eliminar_duplicados_archivo(ruta)

    # =====================
    # RECUPERAR ERRORES
    # =====================

    errores = leer_errores(
        RUTA_ERRORES
    )

    contador = 0

    for error in errores:

        nombre_experimento = error[
            "experimento"
        ]

        frase_error = error[
            "frase"
        ]

        print(
            f"\n[INFO] Recuperando "
            f"{nombre_experimento}"
        )

        parametros = None

        for pset, nombre in EXPERIMENTOS.items():

            if nombre == nombre_experimento:
                parametros = list(pset)
                break

        if parametros is None:

            print(
                f"[WARNING] "
                f"Experimento desconocido: "
                f"{nombre_experimento}"
            )
            continue

        configurar_experimento(
            parametros,
            definiciones_train,
            ejemplos_train,
            ejemplos_eval
        )

        encontrado = False

        for ejemplo, ctx in zip(
            ejemplos_eval,
            CONTEXTOS_POR_FRASE
        ):

            maya = ejemplo[0]
            spa = ejemplo[1]

            if maya != frase_error:
                continue

            encontrado = True

            ruta_salida = (
                f"{RUTA_BASE_RESULTADOS}/"
                f"{nombre_experimento}.txt"
            )

            existentes = {
                r[0]
                for r in leer_resultados(
                    ruta_salida
                )
            }

            if maya in existentes:

                print(
                    "[INFO] Ya existe. "
                    "Saltando."
                )
                break

            prompt = crear_prompt(ctx)

            try:

                respuesta, contador = traducir(
                    client,
                    prompt,
                    contador,
                    nombre_experimento,
                    maya
                )

            except Exception as e:

                print(
                    "[ERROR] "
                    f"No se pudo recuperar:"
                )
                print(e)
                break

            predicciones = parsear_respuesta(
                respuesta
            )

            pred = (
                predicciones[0]
                if predicciones
                else ""
            )

            guardar_resultado_individual(
                ruta_salida,
                maya,
                spa,
                pred
            )

            print(
                "[OK] Recuperada:"
            )
            print(maya)

            break

        if not encontrado:

            print(
                "[WARNING] "
                "Frase no encontrada "
                "en evaluación"
            )

    print(
        "\n[OK] Recuperación completada"
    )

def parsear_respuesta(respuesta):
    lineas = respuesta.strip().split("\n")
    limpias = []

    for linea in lineas:
        linea = linea.strip()

        if "." in linea:
            partes = linea.split(".", 1)
            if partes[0].isdigit():
                linea = partes[1].strip()

        elif ")" in linea:
            partes = linea.split(")", 1)
            if partes[0].isdigit():
                linea = partes[1].strip()

        linea = linea.rstrip(".")

        if linea:
            limpias.append(linea)

    return limpias


def guardar_resultado_individual(
    ruta,
    frase_maya,
    referencia,
    prediccion
):

    with open(ruta, "a", encoding="utf-8") as f:

        f.write("====================================\n")
        f.write(f"MAYA: {frase_maya}\n")
        f.write(f"REF : {referencia}\n")
        f.write(f"PRED: {prediccion}\n")
        f.write("====================================\n\n")

def guardar_error(
    ruta_error,
    nombre_experimento,
    frase_maya,
    error
):

    os.makedirs(os.path.dirname(ruta_error), exist_ok=True)

    with open(ruta_error, "a", encoding="utf-8") as f:

        f.write("====================================\n")
        f.write(f"EXPERIMENTO: {nombre_experimento}\n")
        f.write(f"FRASE      : {frase_maya}\n")
        f.write(f"ERROR      : {str(error)}\n")
        f.write("====================================\n\n")

# =====================
# RECUPERACIÓN
# =====================

def leer_errores(ruta_error):

    if not os.path.exists(ruta_error):
        return []

    with open(ruta_error, "r", encoding="utf-8") as f:
        contenido = f.read()

    patron = re.compile(
        r"EXPERIMENTO:\s*(.*?)\n"
        r"FRASE\s*:\s*(.*?)\n"
        r"ERROR",
        re.DOTALL
    )

    errores = []

    for experimento, frase in patron.findall(contenido):
        errores.append({
            "experimento": experimento.strip(),
            "frase": frase.strip()
        })

    return errores


def leer_resultados(ruta):

    if not os.path.exists(ruta):
        return []

    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = contenido.split(
        "===================================="
    )

    resultados = []

    for bloque in bloques:

        lineas = [
            l.strip()
            for l in bloque.splitlines()
            if l.strip()
        ]

        maya = None
        ref = None
        pred = None

        for linea in lineas:

            if linea.startswith("MAYA:"):
                maya = linea[5:].strip()

            elif linea.startswith("REF :"):
                ref = linea[5:].strip()

            elif linea.startswith("PRED:"):
                pred = linea[5:].strip()

        if maya:
            resultados.append(
                (maya, ref, pred)
            )

    return resultados


def eliminar_duplicados_archivo(ruta):

    resultados = leer_resultados(ruta)

    vistos = set()
    unicos = []

    for maya, ref, pred in resultados:

        if maya in vistos:
            continue

        vistos.add(maya)
        unicos.append(
            (maya, ref, pred)
        )

    with open(ruta, "w", encoding="utf-8") as f:

        for maya, ref, pred in unicos:

            f.write(
                "====================================\n"
            )
            f.write(f"MAYA: {maya}\n")
            f.write(f"REF : {ref}\n")
            f.write(f"PRED: {pred}\n")
            f.write(
                "====================================\n\n"
            )

    print(
        f"[INFO] {os.path.basename(ruta)} "
        f"{len(resultados)} -> {len(unicos)}"
    )

def ejecutar_experimento(
    client,
    nombre_experimento,
    parametros,
    definiciones_train,
    ejemplos_train,
    ejemplos_eval,
    carpeta_salida,
    contador_peticiones,
    thinking=False
):

    sufijo_thinking = "-thinking" if thinking else ""

    ruta_salida = (
        f"{carpeta_salida}/"
        f"{nombre_experimento}{sufijo_thinking}.txt"
    )

    # =====================
    # SALTAR SI YA EXISTE
    # =====================

# =====================
# REANUDAR EXPERIMENTO
# =====================

    num_traducciones = contar_traducciones_realizadas(
        ruta_salida
    )

    total_frases = len(ejemplos_eval)

    if num_traducciones >= total_frases:

        print(f"[INFO] Experimento ya completado:")
        print(f"[INFO] {ruta_salida}")

        return contador_peticiones

    if num_traducciones > 0:

        print(f"[INFO] Reanudando experimento")
        print(f"[INFO] Traducciones existentes: "
            f"{num_traducciones}/{total_frases}")

    else:

        print(f"[INFO] Iniciando experimento nuevo")

    print(f"\n==============================")
    print(f"[INFO] Ejecutando {nombre_experimento}")
    print(f"[INFO] Parámetros: {parametros}")
    print(f"==============================\n")

    configurar_experimento(
        parametros,
        definiciones_train,
        ejemplos_train,
        ejemplos_eval
    )

    ejemplos_pendientes = ejemplos_eval[num_traducciones:]
    contextos_pendientes = CONTEXTOS_POR_FRASE[num_traducciones:]

    for i, (ejemplo, ctx) in enumerate(
        zip(
            ejemplos_pendientes,
            contextos_pendientes
        ),
        start=num_traducciones + 1
    ):

        if contador_peticiones >= MAX_PETICIONES_API:

            print("\n[INFO] Límite de peticiones alcanzado")
            print("[INFO] Terminando ejecución...")

            return contador_peticiones

        frase_maya = ejemplo[0]
        frase_spa = ejemplo[1]

        print(f"\n[INFO] Frase {i}/{len(ejemplos_eval)}")

        prompt = crear_prompt(ctx)

        try:

            respuesta, contador_peticiones = traducir(
                client,
                prompt,
                contador_peticiones,
                nombre_experimento,
                frase_maya,
                thinking=thinking
            )

        except Exception as e:

            print(f"[ERROR] Error traduciendo frase:")
            print(e)

            continue

        predicciones = parsear_respuesta(respuesta)

        prediccion = (
            predicciones[0]
            if predicciones
            else ""
        )

        guardar_resultado_individual(
            ruta_salida,
            frase_maya,
            frase_spa,
            prediccion
        )

        print(f"[OK] Traducción guardada")

        time.sleep(random.uniform(1.0, 2.0))

    print(f"\n[OK] Experimento completado: {nombre_experimento}")

    return contador_peticiones
# ===================== MAIN =====================

def main():

    api_key = open(RUTA_APIKEY).read().strip()

    client = genai.Client(api_key=api_key)

    definiciones_train, ejemplos_train = cargar_datos(
        RUTA_ENTRENAMIENTO
    )

    _, ejemplos_eval = cargar_datos(
        RUTA_EVALUACION
    )

    inicializar_bm25(ejemplos_train)

    os.makedirs(RUTA_BASE_RESULTADOS, exist_ok=True)

    contador_peticiones = 0

    parametros_crudos = sys.argv[1:]

    if "recuperar" in parametros_crudos:
        recuperar_errores(
            client,
            definiciones_train,
            ejemplos_train,
            ejemplos_eval
        )

        return

    # =====================
    # MODO AUTOMÁTICO
    # =====================

    if "all" in parametros_crudos:

        print("\n[INFO] Ejecutando TODOS los experimentos\n")

        for params_set, nombre_experimento in EXPERIMENTOS.items():

            if contador_peticiones >= MAX_PETICIONES_API:

                print("\n[INFO] Límite diario alcanzado")
                return

            parametros = list(params_set)

            contador_peticiones = ejecutar_experimento(
                client,
                nombre_experimento,
                parametros,
                definiciones_train,
                ejemplos_train,
                ejemplos_eval,
                RUTA_BASE_RESULTADOS,
                contador_peticiones,
                thinking=False
            )

        print("\n[INFO] Todos los experimentos completados")
        return

    # =====================
    # MODO NORMAL
    # =====================

    thinking, parametros = separar_parametros_thinking(
        parametros_crudos
    )

    nombre_experimento = obtener_nombre_experimento(
        parametros
    )

    ejecutar_experimento(
        client,
        nombre_experimento,
        parametros,
        definiciones_train,
        ejemplos_train,
        ejemplos_eval,
        RUTA_BASE_RESULTADOS,
        contador_peticiones,
        thinking=thinking
    )

if __name__ == "__main__":
    main()
