#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

BASE_DATASET_DIR="data/datasets/datasets-br-nlp"
LANG_DIR="br"
LOGS_DIR="outputs/logs_slurm_fixed/${LANG_DIR}"
mkdir -p "$LOGS_DIR"

EPOCHS=100
KSHOTS=(1 5 10)
FOLDS=("01" "02" "03" "04" "05")

# Evita resubmeter (e reocupar vaga da QOS onejob) uma combinação que já está
# completa em final_eval_fixed/ — checa ANTES de gerar/submeter o job, não
# depois, pra nem gastar fila com algo que não precisa rodar de novo.
laqda_all_strategies_present() {
    local dir="$1"
    for f in test_final_metrics_report.json test_final_margin_metrics_report.json \
             test_final_mcdropout_metrics_report.json test_final_tempscale_metrics_report.json \
             test_final_xmaha_metrics_report.json; do
        [ -f "${dir}/${f}" ] || return 1
    done
    return 0
}

for CATEGORY in "$BASE_DATASET_DIR"/*; do
  if [ -d "$CATEGORY" ]; then
    for DATASET in "$CATEGORY"/*; do
      if [ -d "$DATASET/few_shot" ]; then
        CORPUS=$(basename "$DATASET")
        DATA_DIR="$DATASET/few_shot"
        FAR_DATA_DIR="$DATASET/few_shot_far_ood"

        for FOLD in ${FOLDS[@]}; do
          for K in ${KSHOTS[@]}; do
            SAVE_DIR_CHECK="outputs/final_eval_fixed/${LANG_DIR}/laqda/${CORPUS}/fold_${FOLD}/kshot_${K}"
            if laqda_all_strategies_present "$SAVE_DIR_CHECK"; then
                echo "✔ LAQDA | ${CORPUS} | Fold ${FOLD} | K-shot ${K} já completo em final_eval_fixed/. Pulando..."
                continue
            fi

            TMP_LAQDA=$(mktemp --suffix=.slurm)
            cat <<EOF > "$TMP_LAQDA"
#!/bin/bash
#SBATCH --job-name=laq_${CORPUS}_${FOLD}_${K}
#SBATCH --output=${LOGS_DIR}/laq_${CORPUS}_${FOLD}_${K}_%j.log
#SBATCH --error=${LOGS_DIR}/laq_${CORPUS}_${FOLD}_${K}_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h100n2,h100n3,b200n1
#SBATCH --gres=gpu:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

SAVE_DIR="outputs/final_eval_fixed/${LANG_DIR}/laqda/${CORPUS}/fold_${FOLD}"
FAR_SAVE_DIR="outputs/final_eval_fixed/${LANG_DIR}/laqda/${CORPUS}_far/fold_${FOLD}"
mkdir -p "\${SAVE_DIR}"

LR="2e-5"
echo "Executando LAQDA | Fold ${FOLD} | K-shot ${K}..."
python -m methods.laqda.cli.train \\
    --dataset_dir "${DATA_DIR}" \\
    --fold "${FOLD}" \\
    --save_dir "\${SAVE_DIR}" \\
    --config configs/methods_config.yaml \\
    --epochs ${EPOCHS} \\
    --kshot "${K}" \\
    --lr \${LR} \\
    --use_sgr

if [ \$? -eq 0 ]; then
    python -m methods.laqda.cli.infer \\
        --dataset_dir "${DATA_DIR}" \\
        --fold "${FOLD}" \\
        --model_paths "\${SAVE_DIR}/kshot_${K}/best_model.pth" \\
        --output_dir "\${SAVE_DIR}" \\
        --kshot "${K}" \\
        --score_mode all
        
    CKPT="\${SAVE_DIR}/kshot_${K}/best_model.pth"
    if [ -d "${FAR_DATA_DIR}/${FOLD}" ]; then
        mkdir -p "\${FAR_SAVE_DIR}"
        echo "Executando LAQDA (far-OOD) | Fold ${FOLD} | K-shot ${K}..."
        python -m methods.laqda.cli.infer \\
            --dataset_dir "${FAR_DATA_DIR}" \\
            --fold "${FOLD}" \\
            --model_paths "\${CKPT}" \\
            --output_dir "\${FAR_SAVE_DIR}" \\
            --kshot "${K}" \\
            --score_mode all
    fi
    
    rm -f "\${SAVE_DIR}/kshot_${K}"/*.pth "\${SAVE_DIR}/kshot_${K}"/*.pt
fi
EOF
            sbatch "$TMP_LAQDA"
            rm -f "$TMP_LAQDA"
          done
        done
      fi
    done
  fi
done
