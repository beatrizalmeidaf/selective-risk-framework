#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

DATA_DIR="data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot"
FOLDS=("01" "02" "03" "04" "05")
K=5
EPOCHS=100
LOGS_DIR="outputs/logs_slurm_fixed/ablation"
mkdir -p "$LOGS_DIR"

# Evita resubmeter (e reocupar vaga da QOS onejob) uma condição/fold que já
# está completa em ablation_fixed/ — checa ANTES de gerar/submeter o job.
ablation_report_present() {
    [ -f "$1/test_final_metrics_report.json" ]
}

for FOLD in ${FOLDS[@]}; do
    for COND in no_sampler no_label_aware prototypes_only; do
        SAVE_DIR_CHECK="outputs/ablation_fixed/IntentPTCorpus/${COND}/fold_${FOLD}/kshot_${K}"
        if ablation_report_present "$SAVE_DIR_CHECK"; then
            echo "✔ Ablação | ${COND} | Fold ${FOLD} já completo em ablation_fixed/. Pulando..."
            continue
        fi

        TMP_ABL=$(mktemp --suffix=.slurm)
        cat <<EOF > "$TMP_ABL"
#!/bin/bash
#SBATCH --job-name=ablation_IntentPT_${COND}_${FOLD}
#SBATCH --output=${LOGS_DIR}/ablation_${COND}_${FOLD}_%j.log
#SBATCH --error=${LOGS_DIR}/ablation_err_${COND}_${FOLD}_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=h100n2,h100n3,b200n1
#SBATCH --gres=gpu:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

case "${COND}" in
    no_sampler)      EXTRA_FLAGS="--disable_transductive_sampler"; INFER_LA="" ;;
    no_label_aware)  EXTRA_FLAGS="--la 0"; INFER_LA="--la 0" ;;
    prototypes_only) EXTRA_FLAGS="--la 0 --disable_transductive_sampler"; INFER_LA="--la 0" ;;
esac

SAVE_DIR="outputs/ablation_fixed/IntentPTCorpus/${COND}/fold_${FOLD}"
mkdir -p "\${SAVE_DIR}"

echo "Executando Ablação | ${COND} | Fold ${FOLD}..."
python -m methods.laqda.cli.train \\
    --dataset_dir "${DATA_DIR}" \\
    --fold "${FOLD}" \\
    --save_dir "\${SAVE_DIR}" \\
    --config configs/methods_config.yaml \\
    --epochs ${EPOCHS} \\
    --kshot ${K} \\
    --lr 2e-5 \\
    --use_sgr \\
    \${EXTRA_FLAGS}

if [ \$? -eq 0 ]; then
    python -m methods.laqda.cli.infer \\
        --dataset_dir "${DATA_DIR}" \\
        --fold "${FOLD}" \\
        --model_paths "\${SAVE_DIR}/kshot_${K}/best_model.pth" \\
        --output_dir "\${SAVE_DIR}" \\
        --kshot ${K} \\
        --score_mode all \\
        \${INFER_LA}
fi

rm -f "\${SAVE_DIR}/kshot_${K}"/*.pth "\${SAVE_DIR}/kshot_${K}"/*.pt
EOF
        sbatch "$TMP_ABL"
        rm -f "$TMP_ABL"
    done
done
