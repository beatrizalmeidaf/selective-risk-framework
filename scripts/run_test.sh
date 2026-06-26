#!/bin/bash
# =============================================================================
# run_test.sh — Smoke test rápido: roda Baseline + LAQDA por 20 épocas
# Dataset: IntentPTCorpus / Fold 01
# Uso: bash scripts/run_test.sh
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

DATA_DIR="data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot"
FOLD="01"
EPOCHS=20
CORPUS="IntentPTCorpus"

echo "=========================================================="
echo "Smoke Test — ${CORPUS} | Fold ${FOLD} | ${EPOCHS} épocas"
echo "=========================================================="
echo ""

# ---------------------------------------------------------------------------
# 1. Job SLURM — Baseline (CrossEntropy) — 20 épocas
# ---------------------------------------------------------------------------
BASELINE_SAVE="outputs/smoke_test/baseline/${CORPUS}/fold_${FOLD}"
mkdir -p "$BASELINE_SAVE"
mkdir -p outputs/logs_slurm

JOB_BASELINE=$(sbatch --parsable \
    --job-name="smoke_baseline" \
    --output="outputs/logs_slurm/smoke_baseline_%j.log" \
    --error="outputs/logs_slurm/smoke_baseline_err_%j.log" \
    --time=00:30:00 \
    --partition=h100n3 \
    --gres=gpu:h100:1 \
    --wrap="
        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
        cd /home/user_beatrizalmeida/selective-risk-framework
        echo '--- Baseline Smoke Test ---'
        echo 'Data: \$(date)'
        python -m methods.baselines.cli.train_baseline \\
            --dataset_dir '${DATA_DIR}' \\
            --fold '${FOLD}' \\
            --save_dir '${BASELINE_SAVE}' \\
            --epochs ${EPOCHS} \\
            --batch_size 16 \\
            --lr 2e-5
        echo 'Baseline concluído: \$(date)'
    "
)

echo "--> [Baseline] Job SLURM submetido: ID=${JOB_BASELINE}"
echo "    Logs: outputs/logs_slurm/smoke_baseline_${JOB_BASELINE}.log"

# ---------------------------------------------------------------------------
# 2. Job SLURM — LAQDA — 20 épocas
# ---------------------------------------------------------------------------
LAQDA_SAVE="outputs/smoke_test/laqda/${CORPUS}/fold_${FOLD}"
mkdir -p "$LAQDA_SAVE"

JOB_LAQDA=$(sbatch --parsable \
    --job-name="smoke_laqda" \
    --output="outputs/logs_slurm/smoke_laqda_%j.log" \
    --error="outputs/logs_slurm/smoke_laqda_err_%j.log" \
    --time=00:30:00 \
    --partition=h100n3 \
    --gres=gpu:h100:1 \
    --wrap="
        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
        cd /home/user_beatrizalmeida/selective-risk-framework
        echo '--- LAQDA Smoke Test ---'
        echo 'Data: \$(date)'
        python -m methods.laqda.cli.train \\
            --dataset_dir '${DATA_DIR}' \\
            --fold '${FOLD}' \\
            --save_dir '${LAQDA_SAVE}' \\
            --config configs/methods_config.yaml \\
            --epochs ${EPOCHS}
        if [ \$? -eq 0 ]; then
            echo 'Treino concluído. Rodando inferência OOD no Test Set...'
            python -m methods.laqda.cli.infer \\
                --dataset_dir '${DATA_DIR}' \\
                --fold '${FOLD}' \\
                --model_paths '${LAQDA_SAVE}/acc_best_model.pth' \\
                --output_dir '${LAQDA_SAVE}'
        fi
        echo 'LAQDA concluído: \$(date)'
    "
)

echo "--> [LAQDA]    Job SLURM submetido: ID=${JOB_LAQDA}"
echo "    Logs: outputs/logs_slurm/smoke_laqda_${JOB_LAQDA}.log"

echo ""
echo "=========================================================="
echo "Jobs submetidos! Para acompanhar:"
echo "  squeue -u \$USER"
echo "  tail -f outputs/logs_slurm/smoke_baseline_${JOB_BASELINE}.log"
echo "  tail -f outputs/logs_slurm/smoke_laqda_${JOB_LAQDA}.log"
echo ""
echo "Resultados salvos em:"
echo "  ${BASELINE_SAVE}/"
echo "  ${LAQDA_SAVE}/"
echo "=========================================================="
