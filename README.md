# Kaqchikel–Spanish Translation Experiments

This repository contains the code used to run the experiments for my Bachelor's Thesis on **Kaqchikel–Spanish machine translation using Gemma 4 and in-context learning**. It includes the complete pipeline for dataset preparation, translation generation, evaluation, and statistical analysis of the experimental results.

## Repository Structure

### `limpiar_JSON.py`
Cleans the original dictionary and removes unnecessary fields, keeping only entries with valid example sentences.  
Run this script first to generate the cleaned dataset.

### `dividir_dataset.py`
Splits the cleaned dictionary into training, evaluation, and static datasets using a fixed random seed.  
Execute it after cleaning the dataset.

### `normalizar_JSON.py`
Normalizes the generated JSON files by standardizing Unicode characters, formatting, and lexical variants.  
Run it after creating the dataset splits.

### `traduccion_kaq_script.py`
Executes all translation experiments with Gemma 4 using the selected retrieval strategies and prompt configurations.  
Configure the desired experiment parameters before running the script.

### `juntar_traducciones.py`
Merges partial translation outputs into a single result file for each experiment.  
Use it once all translations have been generated.

### `evaluar.py`
Computes the evaluation metrics (BLEU, chrF++, TER, and COMET) for every experiment.  
Run it on the merged translation files.

### `analizar_resultados.py`
Performs statistical significance testing and generates plots to compare experimental results.  
Execute it after the evaluation step.

## Execution Order

Run the scripts in the following order:

1. `limpiar_JSON.py`
2. `dividir_dataset.py`
3. `normalizar_JSON.py`
4. `traduccion_kaq_script.py`
5. `juntar_traducciones.py`
6. `evaluar.py`
7. `analizar_resultados.py`