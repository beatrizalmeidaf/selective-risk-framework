#!/bin/bash
# =========================================================================
# Script Unificado EN: Avaliação de todos os algoritmos por 100 épocas
# (Jobs Agrupados por Corpus com Retomada Automática)
# =========================================================================

cd "$(dirname "$0")/.." || exit 1

BASE_DATASET_DIR="data/datasets/datasets-en-nlp"
LANG_DIR="en"
LOGS_DIR="outputs/logs_slurm/${LANG_DIR}"
mkdir -p "$LOGS_DIR"

EPOCHS=100
KSHOTS=(1 5 10)
FOLDS=("01" "02" "03" "04" "05")

echo "=========================================================="
echo "Iniciando Pipeline EN (${EPOCHS} épocas) - Datasets: ${BASE_DATASET_DIR}"
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
        
        if [ -f "\${SAVE_DIR}/kshot_\${K}/test_final_metrics_report.json" ]; then
            echo "✔ Baseline | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        echo "Executando Baseline | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.baselines.cli.train_baseline \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --epochs ${EPOCHS} \\
            --batch_size 16 \\
            --lr 2e-5 \\
            --kshot "\${K}"
    done
done
EOF
        sbatch "$TMP_BASE"
        
        # 2. Job SLURM — KNN-Contrastive
        TMP_KNNC=$(mktemp --suffix=.slurm)
        cat <<EOF > "$TMP_KNNC"
#!/bin/bash
#SBATCH --job-name=knnc_${CORPUS}
#SBATCH --output=${LOGS_DIR}/knnc_${CORPUS}_%j.log
#SBATCH --error=${LOGS_DIR}/knnc_${CORPUS}_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/knn_contrastive/${CORPUS}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"
        
        if [ -f "\${SAVE_DIR}/kshot_\${K}/test_final_metrics_report.json" ]; then
            echo "✔ KNN-Contrastive | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        echo "Executando KNN-Contrastive | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.baselines.cli.train_knn_contrastive \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --epochs ${EPOCHS} \\
            --batch_size 16 \\
            --lr 2e-5 \\
            --kshot "\${K}"
    done
done
EOF
        sbatch "$TMP_KNNC"

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
        
        if [ -f "\${SAVE_DIR}/kshot_\${K}/test_final_metrics_report.json" ]; then
            echo "✔ LAQDA | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        echo "Executando LAQDA | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.laqda.cli.train \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --config configs/methods_config.yaml \\
            --epochs ${EPOCHS} \\
            --kshot "\${K}"
        
        if [ \$? -eq 0 ]; then
            python -m methods.laqda.cli.infer \\
                --dataset_dir "${DATA_DIR}" \\
                --fold "\${FOLD}" \\
                --model_paths "\${SAVE_DIR}/kshot_\${K}/acc_best_model.pth" \\
                --output_dir "\${SAVE_DIR}" \\
                --kshot "\${K}" \\
                --score_mode all
        fi
    done
done
EOF
        sbatch "$TMP_LAQDA"

        # 4. Job SLURM — LAQDA (Com SGR)
        TMP_SGR=$(mktemp --suffix=.slurm)
        cat <<EOF > "$TMP_SGR"
#!/bin/bash
#SBATCH --job-name=sgr_${CORPUS}
#SBATCH --output=${LOGS_DIR}/sgr_${CORPUS}_%j.log
#SBATCH --error=${LOGS_DIR}/sgr_${CORPUS}_err_%j.log
#SBATCH --time=48:00:00
#SBATCH --partition=h100n3
#SBATCH --gres=gpu:h100:1

. /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
cd /home/user_beatrizalmeida/selective-risk-framework

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/laqda_sgr/${CORPUS}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"
        
        if [ -f "\${SAVE_DIR}/kshot_\${K}/test_final_metrics_report.json" ]; then
            echo "✔ LAQDA+SGR | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando..."
            continue
        fi
        
        echo "Executando LAQDA+SGR | Fold \${FOLD} | K-shot \${K}..."
        python -m methods.laqda.cli.train \\
            --dataset_dir "${DATA_DIR}" \\
            --fold "\${FOLD}" \\
            --save_dir "\${SAVE_DIR}" \\
            --config configs/methods_config.yaml \\
            --epochs ${EPOCHS} \\
            --kshot "\${K}" \\
            --use_sgr
        
        if [ \$? -eq 0 ]; then
            python -m methods.laqda.cli.infer \\
                --dataset_dir "${DATA_DIR}" \\
                --fold "\${FOLD}" \\
                --model_paths "\${SAVE_DIR}/kshot_\${K}/acc_best_model.pth" \\
                --output_dir "\${SAVE_DIR}" \\
                --kshot "\${K}" \\
                --score_mode all
        fi
    done
done
EOF
        sbatch "$TMP_SGR"
        
        # Limpa os temporários
        rm -f "$TMP_BASE" "$TMP_KNNC" "$TMP_LAQDA" "$TMP_SGR"
      fi
    done
  fi
done

echo "=========================================================="
echo "Todos os jobs agregados EN submetidos! Acompanhe:"
echo "  squeue -u \$USER"
echo "=========================================================="
