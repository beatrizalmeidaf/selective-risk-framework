#!/bin/bash
# =========================================================================
# Script Unificado PT: Avaliação de todos os algoritmos por 100 épocas
# =========================================================================

cd "$(dirname "$0")/.." || exit 1

BASE_DATASET_DIR="data/datasets/datasets-br-nlp"
LANG_DIR="br"
LOGS_DIR="outputs/logs_slurm/${LANG_DIR}"
mkdir -p "$LOGS_DIR"

EPOCHS=100
KSHOTS=(1 5 10)

echo "=========================================================="
echo "Iniciando Pipeline PT (${EPOCHS} épocas) - Datasets: ${BASE_DATASET_DIR}"
echo "=========================================================="
echo ""

for CATEGORY in "$BASE_DATASET_DIR"/*; do
  if [ -d "$CATEGORY" ]; then
    for DATASET in "$CATEGORY"/*; do
      if [ -d "$DATASET/few_shot" ]; then
        CORPUS=$(basename "$DATASET")
        
        for FOLD_DIR in "$DATASET/few_shot"/*; do
          if [ -d "$FOLD_DIR" ]; then
            FOLD=$(basename "$FOLD_DIR")
            DATA_DIR="$DATASET/few_shot"
            
            for K in "${KSHOTS[@]}"; do
                echo "--> Submetendo jobs para: ${CORPUS} | Fold ${FOLD} | K-shot: ${K}"
                
                # 1. Job SLURM — Baseline
                BASELINE_SAVE="outputs/final_eval/${LANG_DIR}/baseline/${CORPUS}/fold_${FOLD}"
                mkdir -p "$BASELINE_SAVE"
                JOB_BASELINE=$(sbatch --parsable \
                    --job-name="base_${CORPUS}_${FOLD}_${K}" \
                    --output="${LOGS_DIR}/base_${CORPUS}_${FOLD}_${K}_%j.log" \
                    --error="${LOGS_DIR}/base_${CORPUS}_${FOLD}_${K}_err_%j.log" \
                    --time=08:00:00 \
                    --partition=h100n3 \
                    --gres=gpu:h100:1 \
                    --wrap="
                        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
                        cd /home/user_beatrizalmeida/selective-risk-framework
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
                
                # 2. Job SLURM — KNN-Contrastive
                KNNCONT_SAVE="outputs/final_eval/${LANG_DIR}/knn_contrastive/${CORPUS}/fold_${FOLD}"
                mkdir -p "$KNNCONT_SAVE"
                JOB_KNNCONT=$(sbatch --parsable \
                    --job-name="knnc_${CORPUS}_${FOLD}_${K}" \
                    --output="${LOGS_DIR}/knnc_${CORPUS}_${FOLD}_${K}_%j.log" \
                    --error="${LOGS_DIR}/knnc_${CORPUS}_${FOLD}_${K}_err_%j.log" \
                    --time=08:00:00 \
                    --partition=h100n3 \
                    --gres=gpu:h100:1 \
                    --wrap="
                        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
                        cd /home/user_beatrizalmeida/selective-risk-framework
                        python -m methods.baselines.cli.train_knn_contrastive \\
                            --dataset_dir '${DATA_DIR}' \\
                            --fold '${FOLD}' \\
                            --save_dir '${KNNCONT_SAVE}' \\
                            --epochs ${EPOCHS} \\
                            --batch_size 16 \\
                            --lr 2e-5 \\
                            --kshot ${K}
                    "
                )

                # 3. Job SLURM — LAQDA (Sem SGR)
                LAQDA_SAVE="outputs/final_eval/${LANG_DIR}/laqda/${CORPUS}/fold_${FOLD}"
                mkdir -p "$LAQDA_SAVE"
                JOB_LAQDA=$(sbatch --parsable \
                    --job-name="laq_${CORPUS}_${FOLD}_${K}" \
                    --output="${LOGS_DIR}/laq_${CORPUS}_${FOLD}_${K}_%j.log" \
                    --error="${LOGS_DIR}/laq_${CORPUS}_${FOLD}_${K}_err_%j.log" \
                    --time=08:00:00 \
                    --partition=h100n3 \
                    --gres=gpu:h100:1 \
                    --wrap="
                        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
                        cd /home/user_beatrizalmeida/selective-risk-framework
                        python -m methods.laqda.cli.train \\
                            --dataset_dir '${DATA_DIR}' \\
                            --fold '${FOLD}' \\
                            --save_dir '${LAQDA_SAVE}' \\
                            --config configs/methods_config.yaml \\
                            --epochs ${EPOCHS} \\
                            --kshot ${K}
                        
                        if [ \$? -eq 0 ]; then
                            python -m methods.laqda.cli.infer \\
                                --dataset_dir '${DATA_DIR}' \\
                                --fold '${FOLD}' \\
                                --model_paths '${LAQDA_SAVE}/kshot_${K}/acc_best_model.pth' \\
                                --output_dir '${LAQDA_SAVE}' \\
                                --kshot ${K} \\
                                --score_mode all
                        fi
                    "
                )

                # 4. Job SLURM — LAQDA (Com SGR)
                LAQDA_SGR_SAVE="outputs/final_eval/${LANG_DIR}/laqda_sgr/${CORPUS}/fold_${FOLD}"
                mkdir -p "$LAQDA_SGR_SAVE"
                JOB_LAQDA_SGR=$(sbatch --parsable \
                    --job-name="sgr_${CORPUS}_${FOLD}_${K}" \
                    --output="${LOGS_DIR}/sgr_${CORPUS}_${FOLD}_${K}_%j.log" \
                    --error="${LOGS_DIR}/sgr_${CORPUS}_${FOLD}_${K}_err_%j.log" \
                    --time=08:00:00 \
                    --partition=h100n3 \
                    --gres=gpu:h100:1 \
                    --wrap="
                        . /home/user_beatrizalmeida/selective-risk-framework/.venv/bin/activate
                        cd /home/user_beatrizalmeida/selective-risk-framework
                        python -m methods.laqda.cli.train \\
                            --dataset_dir '${DATA_DIR}' \\
                            --fold '${FOLD}' \\
                            --save_dir '${LAQDA_SGR_SAVE}' \\
                            --config configs/methods_config.yaml \\
                            --epochs ${EPOCHS} \\
                            --kshot ${K} \\
                            --use_sgr
                        
                        if [ \$? -eq 0 ]; then
                            python -m methods.laqda.cli.infer \\
                                --dataset_dir '${DATA_DIR}' \\
                                --fold '${FOLD}' \\
                                --model_paths '${LAQDA_SGR_SAVE}/kshot_${K}/acc_best_model.pth' \\
                                --output_dir '${LAQDA_SGR_SAVE}' \\
                                --kshot ${K} \\
                                --score_mode all
                        fi
                    "
                )
            done
          fi
        done
      fi
    done
  fi
done

echo "=========================================================="
echo "Todos os jobs PT submetidos em paralelo! Acompanhe:"
echo "  squeue -u \$USER"
echo "=========================================================="
