#!/bin/bash
# =========================================================================
# Script Unificado: Treinamento LAQDA e Teste de Baselines
# =========================================================================

echo "=========================================================="
echo "Iniciando Pipeline Unificado de Modelos e Baselines"
echo "=========================================================="

# 1. Ativa o ambiente global
source .venv/bin/activate

# 2. Testa as instâncias das classes OOD (Baselines)
echo ""
echo "--> Etapa 1: Validando Algoritmos OOD (MSP, Energy, Mahalanobis, KNN)"
make test-baselines
if [ $? -ne 0 ]; then
    echo "ERRO: Os baselines OOD falharam nos testes unitários."
    exit 1
fi

# 3. Dispara Jobs no SLURM para Todos Datasets e Folds
echo ""
echo "--> Etapa 2: Submetendo Jobs SLURM (Baselines e LAQDA)"

for CATEGORY in data/datasets/datasets-br-nlp/*; do
  if [ -d "$CATEGORY" ]; then
    for DATASET in "$CATEGORY"/*; do
      if [ -d "$DATASET/few_shot" ]; then
        for FOLD_DIR in "$DATASET/few_shot"/*; do
          if [ -d "$FOLD_DIR" ]; then
            FOLD=$(basename "$FOLD_DIR")
            DATA_DIR="$DATASET/few_shot"
            
            # Submete LAQDA (se script existir)
            if [ -f "scripts/run_laqda.slurm" ]; then
                sbatch scripts/run_laqda.slurm "$DATA_DIR" "$FOLD"
            fi
            
            # Submete Baseline
            if [ -f "scripts/run_baseline.slurm" ]; then
                sbatch scripts/run_baseline.slurm "$DATA_DIR" "$FOLD"
            fi
            
          fi
        done
      fi
    done
  fi
done

echo "Jobs SLURM submetidos em paralelo! Acompanhe o progresso em outputs/logs_slurm/"
echo "=========================================================="
echo "Pipeline disparado com sucesso!"
echo "Para avaliar os modelos salvos nos folds, adicione as"
echo "chamadas de inferência neste script conforme necessário."
echo "=========================================================="
