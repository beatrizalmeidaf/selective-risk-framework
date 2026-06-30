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
        
        for i in range(num_folds):
            fold_id = str(i + 1).zfill(2) # "01", "02", "03", "04", "05"
            start_idx = i * fold_size
            end_idx = min((i + 1) * fold_size, total_classes)
            
            ood_classes = all_classes[start_idx:end_idx]
            id_classes = [c for c in all_classes if c not in ood_classes]
            
            splits_dict[corpus_name][fold_id] = {
                "id_classes": id_classes,
                "ood_classes": ood_classes
            }
            
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(splits_dict, f, indent=4, ensure_ascii=False)
        
    print(f"OOD splits determinísticos salvos em {output_file}")

if __name__ == "__main__":
    main()
