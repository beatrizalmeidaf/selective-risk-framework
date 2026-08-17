"""
run_lexical_baseline.py -- Baseline lexical (TF-IDF + regressao logistica) sob o
MESMO protocolo few-shot do pipeline principal.

Motivacao: o ganho de +47 pontos de acuracia no RulingBR (0.468 -> 0.944) e' a
objecao mais previsivel do artigo. A checagem de contaminacao cobre apenas
sobreposicao exata de texto entre particoes, o que nao descarta a hipotese mais
provavel: que o rotulo do RulingBR seja recuperavel de forma quase lexical (a
area do direito citada literalmente no acordao). Um baseline lexical decide
isso -- se TF-IDF com 5 exemplos por classe ja' chega perto, o ganho e' de
atalho lexical e nao de representacao.

Controles: mesmos folds, mesmas classes ID de configs/ood_splits.json, mesmo
suporte K-shot (seed 42, mesma rotina), mesmo conjunto de teste. Confianca e' a
probabilidade maxima da regressao logistica, alimentando as mesmas metricas.

Uso:
    python scripts/llm_baseline/run_lexical_baseline.py --corpus RulingBRCorpus
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.metrics.selective_eval import evaluate_selective, compute_sgr_coverage_at_risk

REPO = Path(__file__).resolve().parents[2]


def find_corpus_dir(corpus):
    hits = [p for p in (REPO / "data").rglob(corpus) if p.is_dir() and (p / "few_shot").is_dir()]
    if not hits:
        raise FileNotFoundError(corpus)
    return hits[0]


def read_jsonl(path):
    out = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(((d.get("text") or d.get("sentence") or ""), d.get("label") or d.get("class_name")))
    return out


def build_support(train_rows, id_classes, kshot, seed=42):
    """Mesma rotina do pipeline principal: seed 42, random.sample por classe."""
    import random
    buckets = {c: [] for c in id_classes}
    for t, l in train_rows:
        if l in buckets:
            buckets[l].append(t)
    random.seed(seed)
    sup = {}
    for c in id_classes:
        s = buckets[c]
        if not s:
            sup[c] = []
        elif len(s) >= kshot:
            sup[c] = random.sample(s, kshot)
        else:
            sup[c] = s * (kshot // len(s)) + s[: kshot % len(s)]
    return sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--folds", nargs="+", default=["01", "02", "03", "04", "05"])
    ap.add_argument("--kshot", type=int, default=5)
    ap.add_argument("--out", default="outputs/lexical_baseline")
    args = ap.parse_args()

    splits = json.load(open(REPO / "configs" / "ood_splits.json"))
    cdir = find_corpus_dir(args.corpus)

    for fold in args.folds:
        t0 = time.time()
        id_classes = splits[args.corpus][fold]["id_classes"]
        cls2idx = {c: i for i, c in enumerate(id_classes)}

        train = read_jsonl(cdir / "few_shot" / fold / "train.jsonl")
        valid = read_jsonl(cdir / "few_shot" / fold / "valid.jsonl")
        test = read_jsonl(cdir / "few_shot" / fold / "test.jsonl")
        sup = build_support(train, id_classes, args.kshot)

        X = [t for c in id_classes for t in sup[c]]
        y = [cls2idx[c] for c in id_classes for _ in sup[c]]
        if len(set(y)) < 2:
            print(f"[skip] fold {fold}: menos de duas classes com suporte")
            continue

        clf = make_pipeline(
            TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=1, max_features=200000),
            LogisticRegression(max_iter=2000, C=10.0, class_weight="balanced"),
        )
        clf.fit(X, y)

        def score(rows):
            P = clf.predict_proba([t for t, _ in rows])
            # remapeia colunas para os indices de classe originais
            full = np.zeros((len(rows), len(id_classes)))
            for j, c in enumerate(clf.classes_):
                full[:, c] = P[:, j]
            return full.max(1), full.argmax(1)

        cal_rows = [(t, l) for t, l in valid if l in cls2idx]
        cal_conf, cal_pred = score(cal_rows)
        cal_tgt = np.array([cls2idx[l] for _, l in cal_rows])

        tst_conf, tst_pred = score(test)
        tst_tgt = np.array([cls2idx.get(l, -1) for _, l in test])

        rep = evaluate_selective(tst_conf, tst_pred, tst_tgt)
        rep.update(compute_sgr_coverage_at_risk(tst_conf, tst_pred, tst_tgt,
                                                calib=(cal_conf, cal_pred, cal_tgt)))
        idm = tst_tgt >= 0
        rep["id_only_accuracy"] = float((tst_pred[idm] == tst_tgt[idm]).mean())
        rep["accuracy"] = float((tst_pred == tst_tgt).mean())
        rep["ood_fraction"] = float((~idm).mean())
        rep["n_support"] = len(X)
        rep["n_test"] = len(test)
        rep["runtime_s"] = round(time.time() - t0, 1)

        od = REPO / args.out / args.corpus / f"fold_{fold}" / f"kshot_{args.kshot}"
        od.mkdir(parents=True, exist_ok=True)
        json.dump(rep, open(od / "lexical_metrics_report.json", "w"), indent=2, ensure_ascii=False)
        print(f"[fold {fold}] acc_id={rep['id_only_accuracy']:.3f} aurc={rep['aurc']:.3f} ({rep['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
