#!/bin/bash
# =============================================================================
# run_test.sh — Smoke test rápido: roda Baseline + LAQDA por 20 épocas
# Dataset: IntentPTCorpus / Fold 01
# Uso: bash scripts/run_test.sh
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

# Extrai active_language da configuracao global
ACTIVE_LANG=$(grep -oP '(?<=active_language:\s")[^"]+' configs/model_encoder_config.yaml || echo "pt")

if [ "$ACTIVE_LANG" == "en" ]; then
    CORPUS="Banking77Corpus"
    DATA_DIR="data/datasets/datasets-en-nlp/intent/${CORPUS}/few_shot"
else
    CORPUS="IntentPTCorpus"
    DATA_DIR="data/datasets/datasets-br-nlp/intent/${CORPUS}/few_shot"
fi

FOLD="01"
EPOCHS=15

echo "=========================================================="
echo "Smoke Test — ${CORPUS} | Fold ${FOLD} | ${EPOCHS} épocas"
echo "=========================================================="
echo ""

# ---------------------------------------------------------------------------
# Loop sobre os K-shots
# ---------------------------------------------------------------------------
KSHOTS=(1 5 10)
for K in "${KSHOTS[@]}"; do
    echo "=========================================================="
    echo "Lançando jobs para K-shot: ${K}"
    echo "=========================================================="

    # 1. Job SLURM — Baseline
    BASELINE_SAVE="outputs/smoke_test/baseline/${CORPUS}/fold_${FOLD}"
    mkdir -p "$BASELINE_SAVE"
    mkdir -p outputs/logs_slurm
    
    JOB_BASELINE=$(sbatch --parsable \
        --job-name="smoke_base_${K}" \
        --output="outputs/logs_slurm/smoke_baseline_${K}_%j.log" \
        --error="outputs/logs_slurm/smoke_baseline_${K}_err_%j.log" \
        --time=04:00:00 \
        --partition=h100n3 \
        --gres=gpu:h100:1 \
        --wrap="
            . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
            cd /home/user_beatrizalmeida/selective-risk-framework
            echo '--- Baseline Smoke Test ---'
            python -m methods.baselines.cli.train_baseline \\
                --dataset_dir '${DATA_DIR}' \\
                --fold '${FOLD}' \\
                --save_dir '${BASELINE_SAVE}' \\
                --epochs ${EPOCHS} \\
                --batch_size 16 \\
                --lr 2e-5 \\
                --kshot ${K}
        "
    )
    echo "--> [Baseline K=${K}] Job SLURM submetido: ID=${JOB_BASELINE}"
    
    # 2. Job SLURM — LAQDA
    LAQDA_SAVE="outputs/smoke_test/laqda/${CORPUS}/fold_${FOLD}"
    mkdir -p "$LAQDA_SAVE"
    
    JOB_LAQDA=$(sbatch --parsable \
        --job-name="smoke_laqda_${K}" \
        --output="outputs/logs_slurm/smoke_laqda_${K}_%j.log" \
        --error="outputs/logs_slurm/smoke_laqda_${K}_err_%j.log" \
        --time=04:00:00 \
        --partition=h100n3 \
        --gres=gpu:h100:1 \
        --wrap="
            . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
            cd /home/user_beatrizalmeida/selective-risk-framework
            echo '--- LAQDA Smoke Test ---'
            python -m methods.laqda.cli.train \\
                --dataset_dir '${DATA_DIR}' \\
                --fold '${FOLD}' \\
                --save_dir '${LAQDA_SAVE}' \\
                --config configs/methods_config.yaml \\
                --epochs ${EPOCHS} \\
                --kshot ${K}
            
            if [ \$? -eq 0 ]; then
                echo 'Treino concluído. Rodando inferência OOD no Test Set...'
                python -m methods.laqda.cli.infer \\
                    --dataset_dir '${DATA_DIR}' \\
                    --fold '${FOLD}' \\
                    --model_paths '${LAQDA_SAVE}/kshot_${K}/acc_best_model.pth' \\
                    --output_dir '${LAQDA_SAVE}' \\
                    --kshot ${K}
            fi
        "
    )
    echo "--> [LAQDA K=${K}] Job SLURM submetido: ID=${JOB_LAQDA}"
    echo ""
done

echo "=========================================================="
echo "Todos os jobs K-shots submetidos! Para acompanhar:"
echo "  squeue -u \$USER"
echo "Resultados serão salvos em:"
echo "  ${BASELINE_SAVE}/kshot_*"
echo "  ${LAQDA_SAVE}/kshot_*"
echo "=========================================================="
