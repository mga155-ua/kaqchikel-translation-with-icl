import os
import re
import sacrebleu

from comet import download_model, load_from_checkpoint


# ==============================
# CARGAR MODELO COMET
# ==============================
print("Cargando modelo COMET...")

model_path = download_model("Unbabel/wmt22-comet-da")
comet_model = load_from_checkpoint(model_path)

print("Modelo COMET cargado.\n")


def cargar_datos(ruta_archivo):

    mayas = []
    referencias = []
    predicciones = []

    with open(ruta_archivo, "r", encoding="utf-8") as f:
        contenido = f.read()

    bloques = contenido.split(
        "===================================="
    )

    for bloque in bloques:

        bloque = bloque.strip()

        if not bloque:
            continue

        maya_match = re.search(
            r"MAYA\s*:\s*(.+)",
            bloque
        )

        ref_match = re.search(
            r"REF\s*:\s*(.+)",
            bloque
        )

        pred_match = re.search(
            r"PRED\s*:\s*(.+)",
            bloque
        )

        if (
            maya_match
            and ref_match
            and pred_match
        ):

            maya = maya_match.group(1).strip()
            referencia = ref_match.group(1).strip()
            prediccion = pred_match.group(1).strip()

            mayas.append(maya)
            referencias.append(referencia)
            predicciones.append(prediccion)

    if not referencias:

        raise ValueError(
            "No se encontraron traducciones válidas"
        )

    if not (
        len(mayas)
        == len(referencias)
        == len(predicciones)
    ):

        raise ValueError(
            "Número distinto de frases entre "
            "MAYA, REF y PRED"
        )

    return (
        mayas,
        referencias,
        predicciones
    )


def evaluar(ruta_archivo):

    srcs, refs, preds = cargar_datos(
        ruta_archivo
    )

    refs_formato = [refs]

    # ==============================
    # MÉTRICAS CLÁSICAS
    # ==============================
    bleu = sacrebleu.corpus_bleu(
        preds,
        refs_formato
    )

    chrf = sacrebleu.corpus_chrf(
        preds,
        refs_formato,
        word_order=2
    )

    ter = sacrebleu.corpus_ter(
        preds,
        refs_formato
    )

    # ==============================
    # COMET
    # ==============================
    comet_data = [
        {
            "src": src,
            "mt": mt,
            "ref": ref
        }
        for src, mt, ref
        in zip(srcs, preds, refs)
    ]

    comet_output = comet_model.predict(
        comet_data,
        batch_size=8,
        gpus=1 if os.environ.get(
            "CUDA_VISIBLE_DEVICES"
        ) else 0
    )

    comet_score = (
        comet_output.system_score
    )

    print("=== RESULTADOS ===")
    print(
        f"Número de frases: "
        f"{len(preds)}\n"
    )

    print(f"BLEU   : {bleu.score:.2f}")
    print(f"chrF++ : {chrf.score:.2f}")
    print(f"TER    : {ter.score:.2f}")
    print(f"COMET  : {comet_score:.4f}")

    return (
        bleu.score,
        chrf.score,
        ter.score,
        comet_score
    )


def obtener_experimentos(
    ruta_txt
):

    with open(
        ruta_txt,
        "r",
        encoding="utf-8"
    ) as f:

        contenido = f.read()

    patrones = re.findall(
        r"experimento(\d+):",
        contenido
    )

    numeros = sorted(
        set(
            int(n)
            for n in patrones
        )
    )

    return numeros


if __name__ == "__main__":

    ruta_experimentos_txt = (
        "../parametros y experimentos.txt"
    )

    carpeta_resultados = (
        "../resultados/resultados_separados/completos"
    )

    experimentos = obtener_experimentos(
        ruta_experimentos_txt
    )

    resultados = []

    for num in experimentos:

        variantes = [
            (
                f"experimento{num}.txt",
                "NORMAL"
            ),
            (
                f"experimento{num}-thinking.txt",
                "THINKING"
            ),
        ]

        for (
            nombre_archivo,
            tipo
        ) in variantes:

            ruta_archivo = os.path.join(
                carpeta_resultados,
                nombre_archivo
            )

            if not os.path.exists(
                ruta_archivo
            ):
                continue

            print(
                f"\n=============================="
            )

            print(
                f"Evaluando: "
                f"{nombre_archivo} ({tipo})"
            )

            print(
                f"=============================="
            )

            try:

                (
                    bleu,
                    chrf,
                    ter,
                    comet
                ) = evaluar(
                    ruta_archivo
                )

                score_total = (
                    bleu
                    + chrf
                    - ter
                )

                resultados.append(
                    {
                        "archivo":
                            nombre_archivo,
                        "tipo":
                            tipo,
                        "bleu":
                            bleu,
                        "chrf":
                            chrf,
                        "ter":
                            ter,
                        "comet":
                            comet,
                        "score":
                            score_total
                    }
                )

            except Exception as e:

                print(
                    f"Error en "
                    f"{nombre_archivo}: {e}"
                )

    # ==============================
    # RANKING CLÁSICO
    # ==============================
    if resultados:

        resultados_ordenados = sorted(
            resultados,
            key=lambda x:
                x["score"],
            reverse=True
        )

        print(
            "\n\nTOP 3 MEJORES EXPERIMENTOS"
        )

        for r in resultados_ordenados[:3]:

            print(
                f"{r['archivo']} "
                f"({r['tipo']}) -> "
                f"Score: {r['score']:.2f} | "
                f"BLEU: {r['bleu']:.2f} | "
                f"chrF++: {r['chrf']:.2f} | "
                f"TER: {r['ter']:.2f} | "
                f"COMET: {r['comet']:.4f}"
            )

        print(
            "\nTOP 2 PEORES EXPERIMENTOS"
        )

        for r in resultados_ordenados[-2:]:

            print(
                f"{r['archivo']} "
                f"({r['tipo']}) -> "
                f"Score: {r['score']:.2f} | "
                f"BLEU: {r['bleu']:.2f} | "
                f"chrF++: {r['chrf']:.2f} | "
                f"TER: {r['ter']:.2f} | "
                f"COMET: {r['comet']:.4f}"
            )

        # ==============================
        # RANKING COMET
        # ==============================
        ranking_comet = sorted(
            resultados,
            key=lambda x:
                x["comet"],
            reverse=True
        )

        print(
            "\n\nTOP 3 POR COMET"
        )

        for r in ranking_comet[:3]:

            print(
                f"{r['archivo']} "
                f"({r['tipo']}) -> "
                f"COMET: {r['comet']:.4f}"
            )

        print(
            "\nPEORES 3 POR COMET"
        )

        for r in ranking_comet[-3:]:

            print(
                f"{r['archivo']} "
                f"({r['tipo']}) -> "
                f"COMET: {r['comet']:.4f}"
            )