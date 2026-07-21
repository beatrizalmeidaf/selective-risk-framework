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
        FAR_DATA_DIR="$DATASET/few_shot_far_ood"

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

# Mesma proteção do bloco LAQDA: exige as 7 estratégias do baseline presentes
# antes de pular, em vez de casar com qualquer uma via glob.
baseline_all_strategies_present() {
    local dir="\$1"
    for f in test_final_knn_metrics_report.json test_final_energy_metrics_report.json \\
             test_final_mahalanobis_metrics_report.json test_final_msp_metrics_report.json \\
             test_final_sota_conjnorm_metrics_report.json test_final_sota_gradnorm_metrics_report.json \\
             test_final_sota_react_metrics_report.json; do
        [ -f "\${dir}/\${f}" ] || return 1
    done
    return 0
}

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/baseline/${CORPUS}/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"

        if baseline_all_strategies_present "\${SAVE_DIR}/kshot_\${K}"; then
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

# "ls test_final_*_metrics_report.json" casava com QUALQUER estratégia presente —
# se só default+margin tivessem sido salvos (ex: mcdropout crashando antes do fix
# do model.train() no MC-Dropout — ver methods/laqda/inference/infer.py), o fold/kshot
# era marcado como "já feito" e nunca mais tentava completar as estratégias que
# faltavam. Exige as 5 estratégias (score_mode=all) presentes antes de pular.
laqda_all_strategies_present() {
    local dir="\$1"
    for f in test_final_metrics_report.json test_final_margin_metrics_report.json \\
             test_final_mcdropout_metrics_report.json test_final_tempscale_metrics_report.json \\
             test_final_xmaha_metrics_report.json; do
        [ -f "\${dir}/\${f}" ] || return 1
    done
    return 0
}

for FOLD in ${FOLDS[@]}; do
    for K in ${KSHOTS[@]}; do
        SAVE_DIR="outputs/final_eval/${LANG_DIR}/laqda/${CORPUS}/fold_\${FOLD}"
        FAR_SAVE_DIR="outputs/final_eval/${LANG_DIR}/laqda/${CORPUS}_far/fold_\${FOLD}"
        mkdir -p "\${SAVE_DIR}"

        NEED_TRAIN=1
        if laqda_all_strategies_present "\${SAVE_DIR}/kshot_\${K}"; then
            echo "✔ LAQDA (near-OOD) | Fold \${FOLD} | K-shot \${K} já treinado e avaliado. Pulando treino..."
            NEED_TRAIN=0
        fi

        if [ "\${NEED_TRAIN}" = "1" ]; then
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
                    --model_paths "\${SAVE_DIR}/kshot_\${K}/best_model.pth" \\
                    --output_dir "\${SAVE_DIR}" \\
                    --kshot "\${K}" \\
                    --score_mode all
            fi
        fi

        # ─── FAR-OOD: reaproveita o MESMO checkpoint, sem retreinar ───
        # Mesmas classes ID / mesmo train-valid (ver scripts/build_far_ood_split.py),
        # só o test.jsonl muda. Roda ANTES do rm -f abaixo, senão o checkpoint some.
        CKPT="\${SAVE_DIR}/kshot_\${K}/best_model.pth"
        if laqda_all_strategies_present "\${FAR_SAVE_DIR}/kshot_\${K}"; then
            echo "✔ LAQDA (far-OOD) | Fold \${FOLD} | K-shot \${K} já avaliado. Pulando..."
        elif [ ! -f "\${CKPT}" ]; then
            echo "⚠ LAQDA (far-OOD) | Fold \${FOLD} | K-shot \${K}: checkpoint ausente (provavelmente já limpo em um run anterior). Precisaria retreinar — pulando."
        elif [ ! -d "${FAR_DATA_DIR}/\${FOLD}" ]; then
            echo "⚠ LAQDA (far-OOD) | Fold \${FOLD}: ${FAR_DATA_DIR}/\${FOLD} não existe. Rode scripts/build_far_ood_split.py antes. Pulando."
        else
            mkdir -p "\${FAR_SAVE_DIR}"
            echo "Executando LAQDA (far-OOD) | Fold \${FOLD} | K-shot \${K}..."
            python -m methods.laqda.cli.infer \\
                --dataset_dir "${FAR_DATA_DIR}" \\
                --fold "\${FOLD}" \\
                --model_paths "\${CKPT}" \\
                --output_dir "\${FAR_SAVE_DIR}" \\
                --kshot "\${K}" \\
                --score_mode all
        fi

        rm -f "\${SAVE_DIR}/kshot_\${K}"/*.pth "\${SAVE_DIR}/kshot_\${K}"/*.pt
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
