#!/bin/bash
#SBATCH --job-name=baseline_test       # Nome do trabalho
#SBATCH --output=outputs/logs_slurm/baseline_saida_%j.log    # Arquivo de saída
#SBATCH --error=outputs/logs_slurm/baseline_erro_%j.log      # Arquivo de erro
#SBATCH --time=05:00:00                # Tempo máximo de execução 
#SBATCH --partition=h100n3             # Partição 
#SBATCH --gres=gpu:h100:1              # Pede UMA GPU

source /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate

echo "=========================================================="
echo "Data de início: $(date)"
echo "Nó de execução: $(hostname)"
echo "GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd /home/user_beatrizalmeida/selective-risk-framework

if [ "$#" -ne 2 ]; then
    echo "Uso: sbatch run_baseline.slurm <dataset_dir> <fold>"
    exit 1
fi

DATA_DIR="$1"
FOLD="$2"
CORPUS=$(basename "$(dirname "$DATA_DIR")")

echo "Rodando Baseline para $CORPUS Fold $FOLD"

SAVE_DIR="outputs/baseline/${CORPUS}/fold_${FOLD}"

mkdir -p "$SAVE_DIR"

python -m methods.baselines.cli.train_baseline \
    --dataset_dir "$DATA_DIR" \
    --fold "$FOLD" \
    --save_dir "$SAVE_DIR" \
    --batch_size 16 \
    --epochs 15 \
    --lr 2e-5

if [ $? -eq 0 ]; then
    echo "--> Treinamento Baseline concluído com sucesso."
else
    echo "ERRO CRÍTICO: Falha no treinamento Baseline. Verifique os logs acima."
fi

echo "=========================================================="
echo "Job Baseline Concluído!"
echo "Data de término: $(date)"
echo "=========================================================="
