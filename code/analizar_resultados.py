import os
import re
import random
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import sacrebleu
from collections import defaultdict


# =========================================
# CONFIGURACIÓN
# =========================================

CARPETA_RESULTADOS = "../resultados/frases_individuales"
RUTA_EXPERIMENTOS = "../parametros y experimentos.txt"

N_BOOTSTRAP = 10000
CONFIDENCE = 0.95
SEED = 42


# =========================================
# CARGAR ARCHIVO DE RESULTADOS
# =========================================

def cargar_datos(ruta_archivo):

    referencias = []
    predicciones = []

    with open(
        ruta_archivo,
        "r",
        encoding="utf-8"
    ) as f:

        ref_actual = None

        for linea in f:

            linea = linea.strip()

            if linea.startswith("REF"):

                ref_actual = (
                    linea
                    .split(":", 1)[1]
                    .strip()
                )

            elif linea.startswith("PRED"):

                pred_actual = (
                    linea
                    .split(":", 1)[1]
                    .strip()
                )

                if ref_actual is not None:

                    referencias.append(
                        ref_actual
                    )

                    predicciones.append(
                        pred_actual
                    )

                    ref_actual = None

    if len(referencias) != len(predicciones):

        raise ValueError(
            f"Número distinto de referencias y "
            f"predicciones en {ruta_archivo}"
        )

    return referencias, predicciones

def alinear_por_referencia(
    refs_a,
    preds_a,
    refs_b,
    preds_b
):

    mapa_b = {
        ref: pred
        for ref, pred in zip(refs_b, preds_b)
    }

    refs_b_ordenadas = []
    preds_b_ordenadas = []

    for ref in refs_a:

        if ref not in mapa_b:

            raise ValueError(
                f"La referencia no existe "
                f"en el segundo archivo:\n{ref}"
            )

        refs_b_ordenadas.append(ref)
        preds_b_ordenadas.append(
            mapa_b[ref]
        )

    if len(mapa_b) != len(refs_a):

        extras = (
            set(mapa_b.keys())
            - set(refs_a)
        )

        raise ValueError(
            "El segundo archivo contiene "
            "referencias adicionales:\n"
            + "\n".join(sorted(extras))
        )

    return (
        refs_a,
        preds_a,
        refs_b_ordenadas,
        preds_b_ordenadas
    )
# =========================================
# CHRF++ POR FRASE
# =========================================

def calcular_chrf_por_frase(referencias, predicciones):

    scores = []

    for ref, pred in zip(referencias, predicciones):

        score = sacrebleu.sentence_chrf(
            hypothesis=pred,
            references=[ref],
            word_order=2
        ).score

        scores.append(score)

    return scores


# =========================================
# OBTENER EXPERIMENTOS
# =========================================

def obtener_experimentos(ruta_txt):

    with open(ruta_txt, "r", encoding="utf-8") as f:
        contenido = f.read()

    patrones = re.findall(r"experimento(\d+):", contenido)

    numeros = sorted(set(int(n) for n in patrones))

    return numeros


# =========================================
# BOOTSTRAP RESAMPLING
# =========================================

def bootstrap_significance(
    scores_a,
    scores_b,
    n_iterations=N_BOOTSTRAP,
    confidence=CONFIDENCE,
    seed=SEED
):

    random.seed(seed)

    differences = []

    n = len(scores_a)

    observed_diff = np.mean(scores_b) - np.mean(scores_a)

    for _ in range(n_iterations):

        indices = [
            random.randint(0, n - 1)
            for _ in range(n)
        ]

        sample_a = [scores_a[i] for i in indices]
        sample_b = [scores_b[i] for i in indices]

        mean_a = np.mean(sample_a)
        mean_b = np.mean(sample_b)

        diff = mean_b - mean_a

        differences.append(diff)

    differences = np.array(differences)

    lower = np.percentile(
        differences,
        ((1 - confidence) / 2) * 100
    )

    upper = np.percentile(
        differences,
        (1 - (1 - confidence) / 2) * 100
    )

    mean_diff = np.mean(differences)

    significant = not (lower <= 0 <= upper)

    # =====================================
    # p-value bootstrap empírico
    # =====================================

    opposite_sign = np.sum(
        np.sign(differences) != np.sign(observed_diff)
    )

    p_value = opposite_sign / n_iterations

    # evitar p=0 exacto

    p_value = max(
        p_value,
        1 / n_iterations
    )

    return {
        "mean_difference": mean_diff,
        "ci_lower": lower,
        "ci_upper": upper,
        "significant": significant,
        "p_value": p_value,
        "std_a": np.std(scores_a),
        "std_b": np.std(scores_b),
        "mean_a": np.mean(scores_a),
        "mean_b": np.mean(scores_b)
    }


# =========================================
# CARGAR TODOS LOS SCORES
# =========================================

def cargar_scores_chrf():

    experimentos = obtener_experimentos(
        RUTA_EXPERIMENTOS
    )

    resultados = defaultdict(list)
    frases_perfectas = defaultdict(list)
    frases_malas = defaultdict(list)

    for num in experimentos:

        variantes = [
            (f"experimento{num}.txt", "NORMAL"),
            (f"experimento{num}-thinking.txt", "THINKING"),
        ]

        for nombre_archivo, tipo in variantes:

            ruta_archivo = os.path.join(
                CARPETA_RESULTADOS,
                nombre_archivo
            )

            if not os.path.exists(ruta_archivo):
                continue

            print(f"Procesando {nombre_archivo}")

            try:

                refs, preds = cargar_datos(
                    ruta_archivo
                )

                key = (
                    f"experimento{num}",
                    tipo
                )

                scores = []

                for ref, pred in zip(refs, preds):

                    score = sacrebleu.sentence_chrf(
                        hypothesis=pred,
                        references=[ref],
                        word_order=2
                    ).score

                    scores.append(score)

                    if score >= 99.999:

                        frases_perfectas[key].append({
                            "referencia": ref,
                            "prediccion": pred
                        })

                    if score < 10:

                        frases_malas[key].append({
                            "referencia": ref,
                            "prediccion": pred,
                            "score": score
                        })

                resultados[key] = scores

            except Exception as e:

                print(
                    f"Error en {ruta_archivo}: {e}"
                )

    return (
    resultados,
    frases_perfectas,
    frases_malas
)


# =========================================
# MOSTRAR RESULTADOS
# =========================================

def imprimir_resultado(exp_a, exp_b, resultado):

    print("\n===================================")
    print(f"{exp_a}  VS  {exp_b}")
    print("===================================")

    print(
        f"\nMEDIA chrF++"
    )

    print(
        f"{exp_a}: "
        f"{resultado['mean_a']:.4f}"
    )

    print(
        f"{exp_b}: "
        f"{resultado['mean_b']:.4f}"
    )

    print(
        f"\nDESVIACIÓN TÍPICA"
    )

    print(
        f"{exp_a}: "
        f"{resultado['std_a']:.4f}"
    )

    print(
        f"{exp_b}: "
        f"{resultado['std_b']:.4f}"
    )

    print(
        f"\nDIFERENCIA MEDIA"
    )

    print(
        f"ΔchrF++: "
        f"{resultado['mean_difference']:.4f}"
    )

    print(
        f"\nINTERVALO DE CONFIANZA"
    )

    print(
        f"IC95%: "
        f"[{resultado['ci_lower']:.4f}, "
        f"{resultado['ci_upper']:.4f}]"
    )

    print(
        f"\nRESULTADO"
    )

    if resultado["significant"]:
        print("Diferencia SIGNIFICATIVA")
    else:
        print("Diferencia NO significativa")

# =========================================
# GUARDAR FRASES PERFECTAS
# =========================================

def guardar_frases_perfectas(frases_perfectas):

    carpeta_salida = os.path.join(
        CARPETA_RESULTADOS,
        "frases_perfectas"
    )

    os.makedirs(
        carpeta_salida,
        exist_ok=True
    )

    ruta_salida = os.path.join(
        carpeta_salida,
        "frases_perfectas.jsonl"
    )

    with open(
        ruta_salida,
        "w",
        encoding="utf-8"
    ) as f:

        for key, ejemplos in frases_perfectas.items():

            experimento, tipo = key

            for dato in ejemplos:

                registro = {
                    "experimento": experimento,
                    "tipo": tipo,
                    "referencia": dato["referencia"],
                    "prediccion": dato["prediccion"],
                    "score": 100.0
                }

                f.write(
                    json.dumps(
                        registro,
                        ensure_ascii=False
                    ) + "\n"
                )

    print(
        "\nFrases perfectas guardadas en:"
    )

    print(ruta_salida)

def guardar_frases_malas(frases_malas):

    carpeta_salida = os.path.join(
        CARPETA_RESULTADOS,
        "frases_perfectas"
    )

    os.makedirs(
        carpeta_salida,
        exist_ok=True
    )

    ruta_salida = os.path.join(
        carpeta_salida,
        "frases_score_menor_10.jsonl"
    )

    with open(
        ruta_salida,
        "w",
        encoding="utf-8"
    ) as f:

        for key, ejemplos in frases_malas.items():

            experimento, tipo = key

            for dato in ejemplos:

                registro = {
                    "experimento": experimento,
                    "tipo": tipo,
                    "referencia": dato["referencia"],
                    "prediccion": dato["prediccion"],
                    "score": dato["score"]
                }

                f.write(
                    json.dumps(
                        registro,
                        ensure_ascii=False
                    ) + "\n"
                )

    print(
        "\nFrases con score < 10 guardadas en:"
    )

    print(ruta_salida)
# =========================================
# HISTOGRAMAS DE DISTRIBUCIÓN
# =========================================

def generar_histogramas(resultados):

    carpeta_salida = os.path.join(
        CARPETA_RESULTADOS,
        "distribucion"
    )

    os.makedirs(
        carpeta_salida,
        exist_ok=True
    )

    for (experimento, tipo), scores in resultados.items():

        if tipo != "NORMAL":
            continue

        plt.figure(figsize=(10, 6))

        plt.hist(
            scores,
            bins=25,
            range=(0, 100),
            color="steelblue",
            edgecolor="black",
            alpha=0.8
        )

        media = np.mean(scores)

        plt.axvline(
            media,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Media = {media:.2f}"
        )

        plt.title(
            f"Distribución chrF++ por frase\n"
            f"{experimento}"
        )

        plt.xlabel("chrF++ por frase")
        plt.ylabel("Frecuencia")

        plt.xlim(0, 100)

        plt.legend()

        nombre_salida = os.path.join(
            carpeta_salida,
            f"{experimento}_histograma.png"
        )

        plt.tight_layout()

        plt.savefig(
            nombre_salida,
            dpi=300
        )

        plt.close()

        print(
            f"Histograma guardado en: "
            f"{nombre_salida}"
        )

# =========================================
# MAIN
# =========================================

# =========================================
# MAIN
# =========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print(
            """
Uso:

Bootstrap entre dos experimentos
--------------------------------
python analizar_resultados.py bootstrap experimento21 experimento22

Bootstrap entre dos archivos concretos
--------------------------------------
python analizar_resultados.py bootstrap experimento21.txt experimento22.txt

Histograma de un experimento
----------------------------
python analizar_resultados.py histograma experimento7

Histogramas de todos
--------------------
python analizar_resultados.py histogramas

Guardar frases perfectas
------------------------
python analizar_resultados.py perfectas

Guardar frases con score chrF++ < 10
-----------------------------
python analizar_resultados.py malas
"""
        )

        sys.exit(0)

    accion = sys.argv[1].lower()

    # =====================================
    # BOOTSTRAP
    # =====================================

    if accion == "bootstrap":

        if len(sys.argv) != 4:

            print(
                "Uso:\n"
                "python analizar_resultados.py "
                "bootstrap experimento21 experimento22"
            )
            sys.exit(1)

        archivo_a = sys.argv[2]
        archivo_b = sys.argv[3]

        if not archivo_a.endswith(".txt"):
            archivo_a += ".txt"

        if not archivo_b.endswith(".txt"):
            archivo_b += ".txt"

        ruta_a = os.path.join(
            CARPETA_RESULTADOS,
            archivo_a
        )

        ruta_b = os.path.join(
            CARPETA_RESULTADOS,
            archivo_b
        )

        refs_a, preds_a = cargar_datos(ruta_a)
        refs_b, preds_b = cargar_datos(ruta_b)

        (
            refs_a,
            preds_a,
            refs_b,
            preds_b
        ) = alinear_por_referencia(
            refs_a,
            preds_a,
            refs_b,
            preds_b
        )

        scores_a = calcular_chrf_por_frase(
            refs_a,
            preds_a
        )

        scores_b = calcular_chrf_por_frase(
            refs_b,
            preds_b
        )

        resultado = bootstrap_significance(
            scores_a,
            scores_b
        )

        imprimir_resultado(
            archivo_a,
            archivo_b,
            resultado
        )

        sys.exit(0)

    # =====================================
    # CARGAR TODOS LOS EXPERIMENTOS
    # =====================================

    resultados, frases_perfectas, frases_malas = (
        cargar_scores_chrf()
    )

    # =====================================
    # HISTOGRAMA DE UN EXPERIMENTO
    # =====================================

    if accion == "histograma":

        if len(sys.argv) != 3:

            print(
                "Uso:\n"
                "python analizar_resultados.py "
                "histograma experimento7"
            )

            sys.exit(1)

        experimento = sys.argv[2]

        key = (experimento, "NORMAL")

        if key not in resultados:

            print(
                f"No existe {experimento}"
            )

            sys.exit(1)

        generar_histogramas(
            {key: resultados[key]}
        )

        sys.exit(0)

    # =====================================
    # HISTOGRAMAS DE TODOS
    # =====================================

    if accion == "histogramas":

        generar_histogramas(resultados)

        sys.exit(0)

    # =====================================
    # FRASES PERFECTAS
    # =====================================

    if accion == "perfectas":

        guardar_frases_perfectas(
            frases_perfectas
        )

        sys.exit(0)

    # =====================================
    # FRASES MALAS
    # =====================================

    if accion == "malas":

        guardar_frases_malas(
            frases_malas
        )

        sys.exit(0)

    print(f"Acción desconocida: {accion}")