"""
run_llm_sgr.py -- Baseline de LLM few-shot com confianca calibravel, avaliada
sob o MESMO protocolo do pipeline principal (mesmos folds, mesmo suporte
K-shot, mesmo split de calibracao, mesmas metricas).

Motivacao: o Apendice "Why no prompted-LLM baseline" argumenta que LLMs sob
prompting nao fornecem uma confianca escalar bem ordenada. Este script testa
essa afirmacao em vez de assumi-la, usando o protocolo de elicitacao mais
defensavel disponivel: multipla escolha com opcoes rotuladas por letra, e
leitura direta dos logits dos tokens-resposta num unico forward pass. Isso
produz uma distribuicao categorica propria sobre as classes -- nao uma
confianca verbalizada -- que pode alimentar o SGR exatamente como o score
geometrico alimenta.

Controles aplicados:
  * exemplos de suporte: mesmos K por classe ID, amostrados do mesmo
    train.jsonl com a mesma seed (42) e a mesma rotina do pipeline principal;
  * classes ID: lidas de configs/ood_splits.json (o mesmo arquivo);
  * calibracao: valid.jsonl restrito a ID, igual ao SGR do paper;
  * metricas: methods/metrics/selective_eval.py, o mesmo codigo -- reportando
    tanto o ponto in-sample (comparavel a Tabela 4) quanto a variante held-out
    (a que carrega a garantia).

Uso:
  python scripts/llm_baseline/run_llm_sgr.py --corpus HateBRCorpus --folds 01 02 03 04 05
"""
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from methods.metrics.selective_eval import evaluate_selective, compute_sgr_coverage_at_risk

REPO = Path(__file__).resolve().parents[2]
# 52 rotulos de token unico no tokenizer do Qwen2.5 (A-Z, a-z), verificado.
# Necessario para corpora de granularidade fina: IntentPT e MMLU-PT-BR tem
# 45-48 classes ID, acima das 26 letras maiusculas.
LETTERS = [chr(ord('A') + i) for i in range(26)] + [chr(ord('a') + i) for i in range(26)]


def find_corpus_dir(corpus: str) -> Path:
    hits = [p for p in (REPO / "data").rglob(corpus) if p.is_dir() and (p / "few_shot").is_dir()]
    if not hits:
        raise FileNotFoundError(f"corpus dir nao encontrado: {corpus}")
    return hits[0]


def read_jsonl(path: Path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            txt = d.get("text") or d.get("sentence") or ""
            lbl = d.get("label") or d.get("class_name")
            out.append((txt, lbl))
    return out


def build_support(train_rows, id_classes, kshot, seed=42):
    """Replica prepare_support_set: seed 42, random.sample por classe, ordem de id_classes."""
    buckets = {c: [] for c in id_classes}
    for txt, lbl in train_rows:
        if lbl in buckets:
            buckets[lbl].append(txt)
    random.seed(seed)
    support = {}
    for c in id_classes:
        s = buckets[c]
        if not s:
            support[c] = []
        elif len(s) >= kshot:
            support[c] = random.sample(s, kshot)
        else:
            support[c] = (s * (kshot // len(s)) + s[: kshot % len(s)])
    return support


def make_prompt(query, id_classes, support, max_chars):
    opts = "\n".join(f"{LETTERS[i]}) {c}" for i, c in enumerate(id_classes))
    parts = [
        "Voce classifica textos em portugues. Responda APENAS com a letra da categoria correta.\n",
        f"Categorias:\n{opts}\n",
        "Exemplos:",
    ]
    # intercala os exemplos por classe para nao enviesar pela ordem
    rounds = max((len(v) for v in support.values()), default=0)
    for r in range(rounds):
        for i, c in enumerate(id_classes):
            if r < len(support[c]):
                parts.append(f"Texto: {support[c][r][:max_chars]}\nResposta: {LETTERS[i]}")
    parts.append(f"Texto: {query[:max_chars]}\nResposta:")
    return "\n\n".join(parts)


@torch.no_grad()
def score_rows(rows, id_classes, support, tok, model, max_chars, batch_size, device):
    """Um forward pass por exemplo; le os logits das letras-resposta."""
    letter_ids = []
    for i in range(len(id_classes)):
        ids = tok.encode(LETTERS[i], add_special_tokens=False)
        letter_ids.append(ids[0])
    letter_ids = torch.tensor(letter_ids, device=device)

    confs, preds = [], []
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [make_prompt(t, id_classes, support, max_chars) for t, _ in chunk]
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}],
                                        tokenize=False, add_generation_prompt=True) for p in prompts]
        # Qwen2.5 aceita 32k de contexto. O limite anterior (8192) truncava pela
        # DIREITA nos corpora de prompt longo (RulingBR: ~8.2-8.6k tokens),
        # removendo o proprio texto a classificar e a deixa "Resposta:" -- o que
        # produzia acuracia ABAIXO do acaso. Truncamos pela esquerda (corta
        # exemplos de suporte, nunca a query) e com folga.
        enc = tok(msgs, return_tensors="pt", padding=True, truncation=True,
                  max_length=32768, padding_side="left").to(device)
        # logits_to_keep=1: sem isso o modelo materializa logits de TODAS as
        # posicoes (batch x seq x 152k vocab), o que estoura 80GB de VRAM em
        # prompts longos.
        out = model(**enc, logits_to_keep=1).logits[:, -1, :]
        sel = out.index_select(1, letter_ids).float()  # (B, n_classes)
        probs = torch.softmax(sel, dim=-1)
        c, p = probs.max(dim=-1)
        confs.extend(c.tolist())
        preds.extend(p.tolist())
        if start % (batch_size * 20) == 0:
            print(f"    {start}/{len(rows)}", flush=True)
    return np.array(confs), np.array(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--folds", nargs="+", default=["01"])
    ap.add_argument("--kshot", type=int, default=5)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--max_chars", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_test", type=int, default=0, help="0 = usar todo o test")
    ap.add_argument("--max_calib", type=int, default=1000)
    ap.add_argument("--out", default="outputs/llm_baseline")
    args = ap.parse_args()

    splits = json.load(open(REPO / "configs" / "ood_splits.json"))
    cdir = find_corpus_dir(args.corpus)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[load] {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # corta exemplos de suporte (inicio), nunca a query e a deixa "Resposta:"
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)
    model.eval()

    for fold in args.folds:
        t0 = time.time()
        id_classes = splits[args.corpus][fold]["id_classes"]
        if len(id_classes) > len(LETTERS):
            print(f"[skip] fold {fold}: {len(id_classes)} classes > {len(LETTERS)} rotulos"); continue

        train = read_jsonl(cdir / "few_shot" / fold / "train.jsonl")
        valid = read_jsonl(cdir / "few_shot" / fold / "valid.jsonl")
        test = read_jsonl(cdir / "few_shot" / fold / "test.jsonl")
        support = build_support(train, id_classes, args.kshot)

        cls2idx = {c: i for i, c in enumerate(id_classes)}
        # calibracao: apenas ID, como no SGR do paper
        cal = [(t, l) for t, l in valid if l in cls2idx][: args.max_calib]
        tst = test if args.max_test == 0 else test[: args.max_test]

        print(f"[fold {fold}] classes={len(id_classes)} calib={len(cal)} test={len(tst)}", flush=True)
        cal_conf, cal_pred = score_rows(cal, id_classes, support, tok, model, args.max_chars, args.batch_size, device)
        cal_tgt = np.array([cls2idx[l] for _, l in cal])

        tst_conf, tst_pred = score_rows(tst, id_classes, support, tok, model, args.max_chars, args.batch_size, device)
        # OOD (classe fora de id_classes) recebe alvo -1, igual ao pipeline
        tst_tgt = np.array([cls2idx.get(l, -1) for _, l in tst])

        rep = evaluate_selective(tst_conf, tst_pred, tst_tgt)
        rep.update(compute_sgr_coverage_at_risk(tst_conf, tst_pred, tst_tgt,
                                                calib=(cal_conf, cal_pred, cal_tgt)))
        id_mask = tst_tgt >= 0
        rep["id_only_accuracy"] = float((tst_pred[id_mask] == tst_tgt[id_mask]).mean())
        rep["accuracy"] = float((tst_pred == tst_tgt).mean())
        rep["ood_fraction"] = float((~id_mask).mean())
        rep["n_test"] = int(len(tst)); rep["n_calib"] = int(len(cal))
        rep["model"] = args.model; rep["kshot"] = args.kshot
        rep["runtime_s"] = round(time.time() - t0, 1)

        od = REPO / args.out / args.corpus / f"fold_{fold}" / f"kshot_{args.kshot}"
        od.mkdir(parents=True, exist_ok=True)
        with open(od / "llm_metrics_report.json", "w") as f:
            json.dump(rep, f, indent=2, ensure_ascii=False)
        print(f"[fold {fold}] acc_id={rep['id_only_accuracy']:.3f} aurc={rep['aurc']:.3f} "
              f"({rep['runtime_s']}s)", flush=True)


if __name__ == "__main__":
    main()
