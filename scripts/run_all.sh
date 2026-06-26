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

# 3. Dispara o LAQDA via SLURM
echo ""
echo "--> Etapa 2: Submetendo Job SLURM para Treinamento do Modelo (LAQDA)"
if [ -f "run_test.slurm" ]; then
    sbatch run_test.slurm
    echo "Job SLURM submetido! Você pode acompanhar o progresso em outputs/logs_slurm/"
else
    echo "AVISO: O arquivo run_test.slurm não foi encontrado na raiz."
fi

echo ""
echo "=========================================================="
echo "Pipeline disparado com sucesso!"
echo "Para avaliar os modelos salvos nos folds, adicione as"
echo "chamadas de inferência neste script conforme necessário."
echo "=========================================================="
