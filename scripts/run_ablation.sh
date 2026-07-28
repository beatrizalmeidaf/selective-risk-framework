#!/bin/bash
# =========================================================================
# Estudo de Ablação — LAQDA (IntentPTCorpus, kshot=5, 5 folds)
#
# Isola os dois componentes reais do LAQDA implementado neste repositório
# (ver methods/laqda/modules/laqda_module.py):
#   - Encoder Label-Aware (cross-attention com a semântica do nome da classe)
#   - TransductiveQDASampler (aumento do protótipo via vizinhos mais
#     próximos da query durante o treino)
#
# Reaproveita os resultados já existentes como referência:
#   - Degrau 1 (baseline + MSP) / Degrau 2 (baseline + melhor score) já
#     estão em outputs/final_eval/br/baseline/IntentPTCorpus/fold_*/kshot_5/
#   - Degrau 4 (LAQDA completo, com os dois componentes) já está em
#     outputs/final_eval/br/laqda/IntentPTCorpus/fold_*/kshot_5/
#
# Este script roda só as 3 condições que faltam:
#   no_sampler        -> label-aware ligado,  sampler desligado
#   no_label_aware     -> label-aware desligado, sampler ligado
#   prototypes_only     -> os dois desligados (rede prototípica pura)
# =========================================================================

cd "$(dirname "$0")/.." || exit 1

DATA_DIR="data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot"
FOLDS=("01" "02" "03" "04" "05")
K=5
EPOCHS=100
LOGS_DIR="outputs/logs_slurm/ablation"
mkdir -p "$LOGS_DIR"

TMP_ABL=$(mktemp --suffix=.slurm)
cat <<EOF > "$TMP_ABL"
#!/bin/bash
#SBATCH --job-name=ablation_IntentPT
#SBATCH --output=${LOGS_DIR}/ablation_%j.log
#SBATCH --error=${LOGS_DIR}/ablation_err_%j.log
#SBATCH --time=24:00:00
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

for FOLD in ${FOLDS[@]}; do
    for COND in no_sampler no_label_aware prototypes_only; do
        case "\${COND}" in
            no_sampler)      EXTRA_FLAGS="--disable_transductive_sampler"; INFER_LA="" ;;
            no_label_aware)  EXTRA_FLAGS="--la 0"; INFER_LA="--la 0" ;;
            prototypes_only) EXTRA_FLAGS="--la 0 --disable_transductive_sampler"; INFER_LA="--la 0" ;;
        esac

        SAVE_DIR="outputs/ablation/IntentPTCorpus/\${COND}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"

        if ls "\${SAVE_DIR}/kshot_${K}"/test_final_metrics_report.json >/dev/null 2>&1; then
            echo "✔ Ablação | \${COND} | Fold \${FOLD} já treinado e avaliado. Pulando..."
            continue
        fi

        echo "Executando Ablação | \${COND} | Fold \${FOLD}..."
        python -m methods.laqda.cli.train \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
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
                --fold "\${FOLD}" \\
                --model_paths "\${SAVE_DIR}/kshot_${K}/best_model.pth" \\
                --output_dir "\${SAVE_DIR}" \\
                --kshot ${K} \\
                --score_mode all \\
                \${INFER_LA}
        fi

        rm -f "\${SAVE_DIR}/kshot_${K}"/*.pth "\${SAVE_DIR}/kshot_${K}"/*.pt
    done
done
EOF

sbatch "$TMP_ABL"
rm -f "$TMP_ABL"

echo "=========================================================="
echo "Job de ablação (IntentPTCorpus, kshot=5) submetido!"
echo "Condições: no_sampler, no_label_aware, prototypes_only"
echo "Resultados em: outputs/ablation/IntentPTCorpus/<condição>/fold_XX/kshot_5/"
echo "Acompanhe: squeue -u \$USER"
echo "=========================================================="
