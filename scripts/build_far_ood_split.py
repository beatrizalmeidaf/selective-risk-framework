"""
build_far_ood_split.py — Gera uma variante FAR-OOD de um dataset few-shot já
existente, sem precisar retreinar nada.

Contexto: os splits padrão em configs/ood_splits.json são NEAR-OOD — as classes
OOD vêm do mesmo corpus (ex: EniacCorpus "geral"/"outros" são categorias
genéricas que compartilham vocabulário com as classes ID, um cenário de
detecção OOD mais difícil por natureza). Este script cria um cenário
complementar FAR-OOD: mesmas classes ID, mesmo train/valid (por isso o
checkpoint já treinado continua válido — não precisa retreinar), mas o
test.jsonl troca as amostras OOD "near" por sentenças de um corpus de domínio
totalmente diferente (ex: decisões jurídicas para um corpus de reviews de
restaurante).

As sentenças far-OOD recebem o label sentinela "__far_ood__", que nunca está
em id_classes — o JSONLDataset (keep_unknown_classes=True) mapeia isso para
class_id=-1 automaticamente, do mesmo jeito que faz com as classes near-OOD.

Uso:
    python scripts/build_far_ood_split.py \\
        --corpus_dir data/datasets/datasets-br-nlp/category/EniacCorpus \\
        --far_source data/datasets/datasets-br-nlp/intent/CourtDecisionCorpus/few_shot/01/train.jsonl \\
        --folds 01 02 03 04 05

Saída: data/datasets/datasets-br-nlp/category/EniacCorpus/few_shot_far_ood/<fold>/{train,valid,test}.jsonl
Idempotente: pula folds cujo test.jsonl já existe, a menos que --force seja passado.
"""
import argparse
import json
import os
import random

FAR_OOD_LABEL = "__far_ood__"


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_text(row):
    return row.get('text') or row.get('sentence')


def get_label(row):
    return row.get('label') or row.get('class_name')


def build_fold(corpus_dir, fold, far_pool, id_classes, seed, force):
    src_dir = os.path.join(corpus_dir, 'few_shot', fold)
    dst_dir = os.path.join(corpus_dir, 'few_shot_far_ood', fold)

    train_src = os.path.join(src_dir, 'train.jsonl')
    valid_src = os.path.join(src_dir, 'valid.jsonl')
    test_src = os.path.join(src_dir, 'test.jsonl')
    test_dst = os.path.join(dst_dir, 'test.jsonl')

    if not os.path.exists(test_src):
        print(f"  [aviso] fold {fold}: {test_src} não encontrado, pulando.")
        return

    if not force and os.path.exists(test_dst):
        print(f"  [skip] fold {fold}: já existe {test_dst}")
        return

    # train/valid: copiados SEM alteração — mesmas classes ID, mesmo suporte
    # few-shot, então o checkpoint treinado no split near-OOD continua válido.
    if os.path.exists(train_src):
        write_jsonl(os.path.join(dst_dir, 'train.jsonl'), read_jsonl(train_src))
    if os.path.exists(valid_src):
        write_jsonl(os.path.join(dst_dir, 'valid.jsonl'), read_jsonl(valid_src))

    test_rows = read_jsonl(test_src)
    id_rows = [r for r in test_rows if get_label(r) in id_classes]
    near_ood_rows = [r for r in test_rows if get_label(r) not in id_classes]
    # Usa a mesma quantidade de amostras OOD do split near-OOD, para manter a
    # fração ID/OOD comparável entre os dois cenários.
    n_far = len(near_ood_rows) if near_ood_rows else max(1, len(id_rows) // 4)

    rs = random.Random(seed)
    sampled_far = rs.sample(far_pool, min(n_far, len(far_pool)))
    far_rows = [{"text": get_text(r), "label": FAR_OOD_LABEL} for r in sampled_far]

    write_jsonl(test_dst, id_rows + far_rows)
    print(f"  [ok] fold {fold}: {len(id_rows)} ID + {len(far_rows)} far-OOD -> {test_dst}")


def main():
    p = argparse.ArgumentParser(description="Gera variante far-OOD de um dataset few-shot (sem retreinar).")
    p.add_argument('--corpus_dir', required=True, help='Ex: data/datasets/datasets-br-nlp/category/EniacCorpus')
    p.add_argument('--far_source', required=True, help='Arquivo .jsonl de domínio distante para servir de OOD')
    p.add_argument('--folds', nargs='+', default=['01', '02', '03', '04', '05'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--force', action='store_true', help='Regenera mesmo se já existir')
    args = p.parse_args()

    corpus_name = os.path.basename(args.corpus_dir.rstrip('/'))
    far_pool = read_jsonl(args.far_source)
    print(f"Fonte far-OOD: {args.far_source} ({len(far_pool)} sentenças disponíveis)")

    splits_file = "configs/ood_splits.json"
    with open(splits_file, encoding='utf-8') as f:
        splits = json.load(f)

    if corpus_name not in splits:
        raise SystemExit(f"Corpus '{corpus_name}' não encontrado em {splits_file}")

    for fold in args.folds:
        if fold not in splits[corpus_name]:
            print(f"  [aviso] fold {fold}: sem split ID/OOD definido em {splits_file}, pulando.")
            continue
        id_classes = set(splits[corpus_name][fold]['id_classes'])
        build_fold(args.corpus_dir, fold, far_pool, id_classes, args.seed + int(fold), args.force)


if __name__ == '__main__':
    main()
