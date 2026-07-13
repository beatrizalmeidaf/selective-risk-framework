#!/bin/bash
# =========================================================================
# Script Unificado PT: Avaliação de todos os algoritmos por 100 épocas
# (Jobs Agrupados por Corpus com Retomada Automática)
# =========================================================================

cd "$(dirname "$0")/.." || exit 1

BASE_DATASET_DIR="data/datasets/datasets-br-nlp"
LANG_DIR="br"
LOGS_DIR="outputs/logs_slurm/${LANG_DIR}"
mkdir -p "$LOGS_DIR"

EPOCHS=100
KSHOTS=(1 5 10)
FOLDS=("01" "02" "03" "04" "05")

echo "=========================================================="
echo "Iniciando Pipeline PT (${EPOCHS} épocas) - Datasets: ${BASE_DATASET_DIR}"
echo "=========================================================="
echo ""

for CATEGORY in "$BASE_DATASET_DIR"/*; do
  if [ -d "$CATEGORY" ]; then
    for DATASET in "$CATEGORY"/*; do
      if [ -d "$DATASET/few_shot" ]; then
        CORPUS=$(basename "$DATASET")
        DATA_DIR="$DATASET/few_shot"
        
        echo "--> Submetendo jobs agregados para: ${CORPUS}"
        
        # 1. Job SLURM — Baseline
        TMP_BASE=$(mktemp --suffix=.slurm)
        cat <<EOF > "$TMP_BASE"
#!/bin/bash
#SBATCH --job-name=base_${CORPUS}
#SBATCH --output=${LOGS_DIR}/base_${CORPUS}_%j.log
#SBATCH --error=${LOGS_DIR}/base_${CORPUS}_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/baseline/${CORPUS}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"
        
        if ls "\${SAVE_DIR}/kshot_\${K}"/test_final_*_metrics_report.json >/dev/null 2>&1; then
            echo "✔ Baseline | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        # Full/Partial Fine-Tuning: necessário para obter bons resultados nos datasets reais
        # Atingir 100% no treino é normal e esperado (overfitting benigno em Few-Shot)
        LR="2e-5"
        EXTRA_ARGS="--num_freeze 6 --weight_decay 0.01 --dropout 0.15"

        echo "Executando Baseline | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.baselines.cli.train_baseline \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --epochs ${EPOCHS} \\
            --batch_size 16 \\
            --lr \${LR} \\
            --kshot "\${K}" \\
            \${EXTRA_ARGS}
    done
done
EOF
        sbatch "$TMP_BASE"
        
        # 3. Job SLURM — LAQDA (Sem SGR)
        TMP_LAQDA=$(mktemp --suffix=.slurm)
        cat <<EOF > "$TMP_LAQDA"
#!/bin/bash
#SBATCH --job-name=laq_${CORPUS}
#SBATCH --output=${LOGS_DIR}/laq_${CORPUS}_%j.log
#SBATCH --error=${LOGS_DIR}/laq_${CORPUS}_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/laqda/${CORPUS}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"
        
        if ls "\${SAVE_DIR}/kshot_\${K}"/test_final_*_metrics_report.json >/dev/null 2>&1; then
            echo "✔ LAQDA | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        LR="2e-5"
        echo "Executando LAQDA | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.laqda.cli.train \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --config configs/methods_config.yaml \\
            --epochs ${EPOCHS} \\
            --kshot "\${K}" \\
            --lr \${LR} \\
            --use_sgr
        
        if [ \$? -eq 0 ]; then
            python -m methods.laqda.cli.infer \\
                --dataset_dir "${DATA_DIR}" \\
                --fold "\${FOLD}" \\
                --model_paths "\${SAVE_DIR}/kshot_\${K}/acc_best_model.pth" \\
                --output_dir "\${SAVE_DIR}" \\
                --kshot "\${K}" \\
                --score_mode all
            rm -f "\${SAVE_DIR}/kshot_\${K}"/*.pth "\${SAVE_DIR}/kshot_\${K}"/*.pt
        fi
    done
done
EOF
        sbatch "$TMP_LAQDA"
        
        # Limpa os temporários
        rm -f "$TMP_BASE" "$TMP_LAQDA"
      fi
    done
  fi
done

echo "=========================================================="
echo "Todos os jobs agregados PT submetidos! Acompanhe:"
echo "  squeue -u \$USER"
echo "=========================================================="
