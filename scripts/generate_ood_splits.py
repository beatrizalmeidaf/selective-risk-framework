import os
import json
import random
import glob
import math

def get_classes_from_jsonl(filepath):
    classes = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if 'label' in data:
                    classes.add(data['label'])
                elif 'class_name' in data:
                    classes.add(data['class_name'])
    return list(classes)

def count_classes_in_train(filepath):
    """Conta exemplos por classe no train.jsonl de UM fold específico (não o global)."""
    counts = {}
    if not os.path.exists(filepath):
        return counts
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            label = data.get('label') or data.get('class_name')
            if label is not None:
                counts[label] = counts.get(label, 0) + 1
    return counts

# Maior kshot usado no pipeline (scripts/run_all_pt.sh: KSHOTS=(1 5 10)). Uma classe
# com menos exemplos de treino que isso num fold específico não sustenta um support
# set real: prepare_support_set (methods/laqda/inference/infer.py) precisaria
# duplicar (kshot > n) ou, no caso extremo de 0 exemplos, falhar — o que antes
# corrompia silenciosamente o alinhamento suporte/query em vez de avisar (ver
# RulingBRCorpus fold 01 "direito financeiro" / fold 02 "direito notarial").
MIN_TRAIN_EXAMPLES = 10

def main():
    base_dirs = ["data/datasets/datasets-br-nlp", "data/datasets/datasets-en-nlp"]
    output_file = "configs/ood_splits.json"
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    splits_dict = {}
    
    fold_1_files = []
    for base_dir in base_dirs:
        # Busca por todas as categorias de datasets recursivamente.
        # Estrutura esperada: data/datasets/<idioma>/<categoria>/<nome_corpus>/few_shot/<fold>/train.jsonl
        search_pattern = os.path.join(base_dir, "*", "*", "few_shot", "01", "train.jsonl")
        fold_1_files.extend(glob.glob(search_pattern))
    
    print(f"Encontrados {len(fold_1_files)} datasets.")
    
    for train_file in fold_1_files:
        # Extrai o nome do corpus a partir do caminho do arquivo.
        parts = train_file.split(os.sep)
        # O diretório base geralmente fica nos primeiros índices.
        # Como o caminho termina em ... / <nome_corpus> / few_shot / 01 / train.jsonl
        # O nome do corpus sempre será o 4º elemento contando de trás para frente.
        corpus_name = parts[-4]
        
        # Precisamos de todas as classes deste corpus. 
        # Lemos os arquivos de treino, validação e teste do fold 01 para garantir que não falta nenhuma classe.
        fold_1_dir = os.path.dirname(train_file)
        all_classes = set()
        for split in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
            split_path = os.path.join(fold_1_dir, split)
            if os.path.exists(split_path):
                all_classes.update(get_classes_from_jsonl(split_path))
                
        all_classes = sorted(list(all_classes))
        
        # Embaralhamento determinístico para garantir reprodutibilidade.
        random.seed(42)
        random.shuffle(all_classes)
        
        total_classes = len(all_classes)
        # Criação de 5 divisões (folds) sobre as classes (20% das classes serão OOD por fold).
        num_folds = 5
        fold_size = math.ceil(total_classes / num_folds)

        splits_dict[corpus_name] = {}

        # Corpora com < 3 classes (ex: binários de sentimento/hate: Ofensivo vs
        # Não Ofensivo) não suportam holdout de classe: reter 1 classe como OOD já
        # deixa só 1 classe ID, tornando a classificação trivial (só existe uma
        # resposta possível -> 100% de acurácia garantido, não é o modelo aprendendo
        # nada). Nesses casos mantemos todas as classes como ID nos 5 folds; a
        # avaliação de OOD desses corpora passa a depender só do far-OOD
        # (few_shot_far_ood/build_far_ood_split.py), não do holdout por classe aqui.
        if total_classes < 3:
            print(f"  [Aviso] {corpus_name}: {total_classes} classe(s) totais — "
                  f"holdout por classe desativado (usar apenas far-OOD para este corpus).")
            for i in range(num_folds):
                fold_id = str(i + 1).zfill(2)
                splits_dict[corpus_name][fold_id] = {
                    "id_classes": all_classes,
                    "ood_classes": []
                }
            continue

        for i in range(num_folds):
            fold_id = str(i + 1).zfill(2) # "01", "02", "03", "04", "05"
            start_idx = i * fold_size
            end_idx = min((i + 1) * fold_size, total_classes)

            ood_classes = all_classes[start_idx:end_idx]
            id_classes = [c for c in all_classes if c not in ood_classes]

            # Classes raras (cauda longa) podem ter poucos ou zero exemplos NUM FOLD
            # ESPECÍFICO mesmo aparecendo em all_classes (que veio de train+valid+test
            # do fold 01 apenas). Reclassificamos aqui usando o train.jsonl real DESTE
            # fold: id_classes sem exemplos suficientes viram ood_classes — o mesmo
            # destino que StandardDataModule já dá a qualquer classe fora de
            # id_classes (class_id=-1), então isso não muda o pipeline, só evita
            # prometer uma classe ID que não tem como virar support set.
            fold_train_path = os.path.join(os.path.dirname(fold_1_dir), fold_id, "train.jsonl")
            fold_counts = count_classes_in_train(fold_train_path)

            reassigned = [c for c in id_classes if fold_counts.get(c, 0) < MIN_TRAIN_EXAMPLES]
            if reassigned:
                id_classes = [c for c in id_classes if c not in reassigned]
                ood_classes = ood_classes + reassigned
                for c in reassigned:
                    print(f"  [Aviso] {corpus_name} fold {fold_id}: classe '{c}' tem "
                          f"{fold_counts.get(c, 0)} exemplo(s) de treino (< {MIN_TRAIN_EXAMPLES}) "
                          f"— movida de id_classes para ood_classes.")

            if len(id_classes) < 2:
                print(f"  [Aviso] {corpus_name} fold {fold_id}: só sobrou(ram) "
                      f"{len(id_classes)} classe(s) ID depois de remover as raras — "
                      f"classificação vai ficar trivial/degenerada nesse fold. "
                      f"Revisar manualmente.")

            splits_dict[corpus_name][fold_id] = {
                "id_classes": id_classes,
                "ood_classes": ood_classes
            }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(splits_dict, f, indent=4, ensure_ascii=False)
        
    print(f"OOD splits determinísticos salvos em {output_file}")

if __name__ == "__main__":
    main()
