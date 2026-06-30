#!/bin/bash
#SBATCH --job-name=selective_laqda_test          # Nome do trabalho
#SBATCH --output=outputs/logs_slurm/selective_laqda_saida_%j.log    # Arquivo de saída
#SBATCH --error=outputs/logs_slurm/selective_laqda_erro_%j.log      # Arquivo de erro
#SBATCH --time=09:00:00                # Tempo máximo de execução 
#SBATCH --partition=h100n3             # Partição 
#SBATCH --gres=gpu:h100:1              # Pede UMA GPU (conforme solicitado para o teste)

source /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate

echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd /home/user_beatrizalmeida/selective-risk-framework

echo "Rodando Selective LAQDA..."

if [ "$#" -ne 2 ]; then
    echo "Uso: sbatch run_laqda.slurm <dataset_dir> <fold>"
    exit 1
fi

DATA_DIR="$1"
FOLD="$2"
CORPUS=$(basename "$(dirname "$DATA_DIR")")

SAVE_DIR="outputs/laqda/${CORPUS}/fold_${FOLD}"
mkdir -p "$SAVE_DIR"

python -m methods.laqda.cli.train \
    --dataset_dir "$DATA_DIR" \
    --fold "$FOLD" \
    --save_dir "$SAVE_DIR" \
    --config methods/laqda/configs/default.yaml

if [ $? -eq 0 ]; then
    echo "--> Fold $FOLD LAQDA Train concluído. Iniciando Avaliação OOD (Test Set)..."
    python -m methods.laqda.cli.infer \
        --dataset_dir "$DATA_DIR" \
        --fold "$FOLD" \
        --model_paths "$SAVE_DIR/acc_best_model.pth" \
        --output_dir "$SAVE_DIR"
else
    echo "ERRO CRÍTICO: Falha no Fold $FOLD do LAQDA."
fi

echo "=========================================================="
echo "Job LAQDA Concluído!"
echo "Data de término: $(date)"
echo "=========================================================="
