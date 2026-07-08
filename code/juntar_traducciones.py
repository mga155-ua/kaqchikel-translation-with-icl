import os
import re

CARPETA_PC1 = "../resultados/resultados_separados/frases_individuales1"
CARPETA_PC2 = "../resultados/resultados_separados/frases_individuales2"
CARPETA_SALIDA = "../resultados/resultados_separados/completos"

NUM_EXPERIMENTOS = 22


def leer_resultados(ruta):

    with open(ruta, "r", encoding="utf-8") as f:
        contenido = f.read()

    patron = re.compile(
        r"MAYA:\s*(.*?)\n"
        r"REF\s*:\s*(.*?)\n"
        r"PRED:\s*(.*?)\n",
        re.DOTALL
    )

    resultados = []

    for maya, ref, pred in patron.findall(contenido):

        resultados.append({
            "maya": maya.strip(),
            "ref": ref.strip(),
            "pred": pred.strip()
        })

    return resultados


def unir_archivos(
    archivo1,
    archivo2,
    archivo_salida
):

    resultados_1 = leer_resultados(archivo1)
    resultados_2 = leer_resultados(archivo2)

    todos = resultados_1 + resultados_2

    unicos = {}
    duplicados = []

    for resultado in todos:

        maya = resultado["maya"]

        if maya in unicos:
            duplicados.append(maya)
        else:
            unicos[maya] = resultado

    print("\n========================")
    print(os.path.basename(archivo_salida))
    print("========================")

    print(f"Frases archivo 1: {len(resultados_1)}")
    print(f"Frases archivo 2: {len(resultados_2)}")
    print(f"Frases únicas: {len(unicos)}")
    print(f"Duplicados encontrados: {len(duplicados)}")

    with open(
        archivo_salida,
        "w",
        encoding="utf-8"
    ) as f:

        for resultado in unicos.values():

            f.write(
                "====================================\n"
            )

            f.write(
                f"MAYA: {resultado['maya']}\n"
            )

            f.write(
                f"REF : {resultado['ref']}\n"
            )

            f.write(
                f"PRED: {resultado['pred']}\n"
            )

            f.write(
                "====================================\n\n"
            )

    print(f"[OK] Generado: {archivo_salida}")


if __name__ == "__main__":

    os.makedirs(
        CARPETA_SALIDA,
        exist_ok=True
    )

    for i in range(1, NUM_EXPERIMENTOS + 1):

        archivo1 = os.path.join(
            CARPETA_PC1,
            f"experimento{i}.txt"
        )

        archivo2 = os.path.join(
            CARPETA_PC2,
            f"experimento{i}.txt"
        )

        archivo_salida = os.path.join(
            CARPETA_SALIDA,
            f"experimento{i}.txt"
        )

        if not os.path.exists(archivo1):
            print(f"[ERROR] No existe: {archivo1}")
            continue

        if not os.path.exists(archivo2):
            print(f"[ERROR] No existe: {archivo2}")
            continue

        unir_archivos(
            archivo1,
            archivo2,
            archivo_salida
        )

    print("\nTodos los experimentos procesados.")