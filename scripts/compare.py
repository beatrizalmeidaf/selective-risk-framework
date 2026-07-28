"""
compare.py — Script unificado de comparação de métricas do Selective Risk Framework (com suporte a agregação Cross-Fold e Gráficos).

Modos de uso:
  # Tabela agregada completa e gráfico para um corpus:
  python scripts/compare.py --corpus IntentPTCorpus --plot

  # Tabela rápida:
  python scripts/compare.py --corpus IntentPTCorpus --mode fast

  # Filtrar um kshot específico:
  python scripts/compare.py --corpus IntentPTCorpus --kshot 10

  # Salvar resultado em CSV:
  python scripts/compare.py --corpus IntentPTCorpus --save outputs/comparison.csv
"""

import argparse
import json
import os
import glob
import sys
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOT_LIBS = True
except ImportError:
    HAS_PLOT_LIBS = False


# ──────────────────────────────────────────────────────────────────────────────
# Configuração dos métodos disponíveis
# ──────────────────────────────────────────────────────────────────────────────

METHODS = [
    ("base_msp",        "baseline",  "msp"),
    ("base_energy",     "baseline",  "energy"),
    ("base_mahal",      "baseline",  "mahalanobis"),
    ("base_knn",        "baseline",  "knn"),
    ("base_conjnorm",   "baseline",  "sota_conjnorm"),
    ("base_gradnorm",   "baseline",  "sota_gradnorm"),
    ("base_react",      "baseline",  "sota_react"),
    ("laqda",           "laqda",     ""),
    ("laqda_margin",    "laqda",     "margin"),
    ("laqda_mcdrop",    "laqda",     "mcdropout"),
    ("laqda_tempscale", "laqda",     "tempscale"),
    ("laqda_xmaha",     "laqda",     "xmaha"),
]

KSHOTS   = ["kshot_1", "kshot_5", "kshot_10"]
_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# Baseline não tem o bug de gradient checkpointing (use_reentrant=True zerava o
# gradiente das camadas destravadas do BERT — ver methods/laqda/models/encoder.py),
# então continua lido de outputs/final_eval/ (não precisa re-treinar). LAQDA, com
# o fix, grava numa árvore separada (outputs/final_eval_fixed/) para nunca
# misturar com os resultados antigos pré-fix — ver scripts/run_all_pt.sh.
BASE_DIR_BASELINE = os.path.join(_ROOT, "outputs", "final_eval")
BASE_DIR_LAQDA = os.path.join(_ROOT, "outputs", "final_eval_fixed")
BASE_DIRS = {"baseline": BASE_DIR_BASELINE, "laqda": BASE_DIR_LAQDA}
REPORTS_DIR = os.path.join(_ROOT, "outputs", "reports")

SORT_KEYS = {
    "cov10":   "sgr_coverage_at_risk_10",
    "cov5":    "sgr_coverage_at_risk_5",
    "cov1":    "sgr_coverage_at_risk_1",
    "aurc":    "aurc",
    "acc":     "accuracy",
    "id_acc":  "id_only_accuracy",
    "auroc":   "auroc",
    "f1":      "f1_macro",
}


# ──────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ──────────────────────────────────────────────────────────────────────────────

METADATA_CACHE = {}

def resolve_dataset_metadata(corpus: str):
    """Inferir Idioma e Categoria instantaneamente a partir do nome do corpus."""
    if corpus in METADATA_CACHE:
        return METADATA_CACHE[corpus]

    base_corpus = corpus[:-len("_far")] if corpus.endswith("_far") else corpus
    is_pt = any(substr in base_corpus for substr in ["PT", "BR", "TuPy", "Eniac", "Court", "Hate", "Olist", "RePro", "Recognasumm", "Buscape", "B2W", "Brands", "Ruling", "UTL"])
    lang = "PT" if is_pt else "EN"
    cat = "Intent" if "Intent" in base_corpus or "Snips" in base_corpus or "CLINC" in base_corpus or "Banking" in base_corpus else "Classification"
    res = (lang, cat)
    METADATA_CACHE[corpus] = res
    return res


REPORT_CACHE = {}
FS_CACHE = {}

def build_fs_cache(base_path: str):
    # Indexado por base_path: baseline (final_eval/) e laqda (final_eval_fixed/)
    # são árvores DIFERENTES agora, então cada uma precisa do seu próprio índice
    # — um cache global único (como antes) faria a segunda árvore nunca ser
    # escaneada.
    if base_path in FS_CACHE:
        return
    cache = set()
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".json"):
                cache.add(os.path.abspath(os.path.join(root, f)))
    FS_CACHE[base_path] = cache

def load_report_aggregated(base_path: str, lang_dir: str, folder: str, corpus: str, kshot: str, suffix: str):
    """Carrega os resultados de múltiplos folds (01 a 05) e calcula a média e o desvio padrão com caching."""
    cache_key = (base_path, lang_dir, folder, corpus, kshot, suffix)
    if cache_key in REPORT_CACHE:
        return REPORT_CACHE[cache_key]

    build_fs_cache(base_path)
    folds = [f"fold_0{i}" for i in range(1, 6)]
    
    metrics = {
        "accuracy": [], "id_only_accuracy": [], "ood_fraction": [],
        "f1_macro": [], "aurc": [], "e_aurc": [],
        "auroc": [], "sgr_coverage_at_risk_1": [], "sgr_coverage_at_risk_5": [],
        "sgr_coverage_at_risk_10": [], "fpr_at_95": [], "ece": [],
        "sgr_accepted_accuracy": [], "sgr_abstention_rate": [],
        "sgr_id_coverage": [], "sgr_ood_rejection_rate": []
    }
    
    folds_found = 0
    for fold in folds:
        fname = "test_final_metrics_report.json" if suffix == "" else f"test_final_{suffix}_metrics_report.json"
        path = os.path.abspath(os.path.join(base_path, lang_dir, folder, corpus, fold, kshot, fname))
        
        if path in FS_CACHE.get(base_path, ()):
            try:
                with open(path) as f:
                    d = json.load(f)
                    for k in metrics.keys():
                        if k in d and d[k] is not None:
                            metrics[k].append(d[k])
                folds_found += 1
            except Exception:
                pass
            
    if folds_found == 0:
        REPORT_CACHE[cache_key] = None
        return None
        
    aggregated = {"folds_found": folds_found}
    for k, vlist in metrics.items():
        if len(vlist) > 0:
            aggregated[k] = (np.mean(vlist), np.std(vlist))
        else:
            aggregated[k] = (None, None)
            
    REPORT_CACHE[cache_key] = aggregated
    return aggregated


def collect_rows_aggregated(base_dir_map: dict, lang_dir: str, corpus: str, kshot: str, sort_key: str, asc: bool):
    """Coleta e ordena as linhas agregadas para todos os métodos.

    base_dir_map: {"baseline": <dir>, "laqda": <dir>} — cada método pode viver
    numa árvore de outputs diferente (ver BASE_DIRS acima).
    """
    metric_field = SORT_KEYS.get(sort_key, "sgr_coverage_at_risk_10")
    rows = []

    for label, folder, suffix in METHODS:
        d = load_report_aggregated(base_dir_map[folder], lang_dir, folder, corpus, kshot, suffix)
        if d is None:
            continue
            
        rows.append({
            "label":     label,
            "folder":    folder,
            "folds":     d["folds_found"],
            "acc":       d.get("accuracy"),
            "id_acc":    d.get("id_only_accuracy"),
            "ood_frac":  d.get("ood_fraction"),
            "f1":        d.get("f1_macro"),
            "aurc":      d.get("aurc"),
            "e_aurc":    d.get("e_aurc"),
            "auroc":     d.get("auroc"),
            "cov1":      d.get("sgr_coverage_at_risk_1"),
            "cov5":      d.get("sgr_coverage_at_risk_5"),
            "cov10":     d.get("sgr_coverage_at_risk_10"),
            "fpr95":     d.get("fpr_at_95"),
            "ece":       d.get("ece"),
            "sgr_acc":   d.get("sgr_accepted_accuracy"),
            "sgr_abs":   d.get("sgr_abstention_rate"),
            "sgr_idcov": d.get("sgr_id_coverage"),
            "sgr_oodrej": d.get("sgr_ood_rejection_rate"),
        })
        
    def sort_func(x):
        val_tuple = x.get(sort_key)
        if val_tuple is None or val_tuple[0] is None:
            return 0.0
        return val_tuple[0]
        
    rows.sort(key=sort_func, reverse=(not asc))
    return rows


def fmt_agg(v_tuple, width=15):
    """Formata tupla (mean, std) para exibição."""
    if v_tuple is None or v_tuple[0] is None:
        return " " * (width - 1) + "—"
    mean, std = v_tuple
    s = f"{mean:.4f} ± {std:.4f}"
    return f"{s:>{width}}"


def marker(folder: str) -> str:
    """Símbolo de destaque por tipo de método."""
    if "laqda" in folder:
        return "★"
    return " "


# ──────────────────────────────────────────────────────────────────────────────
# Modos de exibição e Gráficos
# ──────────────────────────────────────────────────────────────────────────────

def print_full(rows, kshot: str):
    """Tabela completa com todas as métricas agregadas.

    'Acc(ID+OOD)' é limitada estruturalmente por (1 - ood_fraction): amostras OOD
    nunca podem ser acertadas por um classificador fechado, então essa coluna NÃO
    mede a qualidade real do classificador — só serve de contexto. Use
    'Acc(ID)' (id_only_accuracy) para comparar classificação, e AUROC/FPR@95 para
    comparar detecção OOD (ver docs/metrics.md).

    'SGR@5%'/'SGR@10%' (sgr_coverage_at_risk_*) exigem garantir risco <= r* no
    conjunto de teste MISTURADO com OOD (PAC bound) — costuma zerar quando o AUROC
    é fraco, mesmo com boa acurácia ID. 'SGR IDCov'/'SGR OODRej' (sgr_id_coverage /
    sgr_ood_rejection_rate) são a métrica complementar: o threshold é calibrado só
    em ID (val), aplicado ao teste, e reporta separadamente quanto do ID continua
    sendo servido vs. quanto do OOD é corretamente rejeitado — normalmente mais
    informativa que o SGR@ zerado.
    """
    W = 230
    print(f"\n{'─' * W}")
    print(f"  {kshot.upper()} (Médias sobre múltiplos folds)")
    print(f"{'─' * W}")
    header = (
        f"  {'Método':<22} | {'Folds'} | {'Acc(ID+OOD)':>15} | {'Acc(ID)':>15} | {'AURC':>15} | "
        f"{'E-AURC':>15} | {'SGR@5%':>15} | {'SGR@10%':>15} | "
        f"{'SGR IDCov':>15} | {'SGR OODRej':>15} | {'AUROC':>15} | {'FPR@95':>15}"
    )
    print(header)
    print(f"{'─' * W}")
    for r in rows:
        m = marker(r["folder"])
        print(
            f"{m} {r['label']:<22} |   {r['folds']}   | {fmt_agg(r['acc'])} | {fmt_agg(r['id_acc'])} | {fmt_agg(r['aurc'])} | "
            f"{fmt_agg(r['e_aurc'])} | {fmt_agg(r['cov5'])} | {fmt_agg(r['cov10'])} | "
            f"{fmt_agg(r['sgr_idcov'])} | {fmt_agg(r['sgr_oodrej'])} | "
            f"{fmt_agg(r['auroc'])} | {fmt_agg(r['fpr95'])}"
        )
    if rows and rows[0].get("ood_frac") and rows[0]["ood_frac"][0] is not None:
        print(f"  (fração OOD do split: {rows[0]['ood_frac'][0]*100:.1f}% -> teto de Acc(ID+OOD) = "
              f"{(1 - rows[0]['ood_frac'][0])*100:.1f}%, mesmo com classificador perfeito nas classes ID)")


def print_fast(rows, kshot: str):
    """Tabela rápida com métricas principais."""
    W = 140
    print(f"\n{'─' * W}")
    print(f"  {kshot.upper()} (Médias sobre múltiplos folds)")
    print(f"{'─' * W}")
    header = f"  {'Método':<22} | {'Acc(ID)':>15} | {'AURC':>15} | {'SGR@10%':>15} | {'SGR IDCov':>15} | {'SGR OODRej':>15}"
    print(header)
    print(f"{'─' * W}")
    for r in rows:
        m = marker(r["folder"])
        print(
            f"{m} {r['label']:<22} | {fmt_agg(r['id_acc'])} | {fmt_agg(r['aurc'])} | {fmt_agg(r['cov10'])} | "
            f"{fmt_agg(r['sgr_idcov'])} | {fmt_agg(r['sgr_oodrej'])}"
        )


def print_markdown(rows, kshot: str):
    """Tabela completa em formato Markdown."""
    print(f"\n### {kshot.upper()} (Médias sobre múltiplos folds)\n")
    
    header = "| Método | Folds | Acc(ID+OOD) | Acc(ID) | AURC | E-AURC | SGR@5% | SGR@10% | SGR IDCov | SGR OODRej | AUROC | FPR@95 |"
    print(header)
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    
    def f(v):
        return fmt_agg(v, width=0).strip()
        
    for r in rows:
        m = marker(r["folder"])
        print(
            f"| {m} {r['label']} | {r['folds']} | {f(r['acc'])} | {f(r['id_acc'])} | {f(r['aurc'])} | "
            f"{f(r['e_aurc'])} | {f(r['cov5'])} | {f(r['cov10'])} | "
            f"{f(r['sgr_idcov'])} | {f(r['sgr_oodrej'])} | "
            f"{f(r['auroc'])} | {f(r['fpr95'])} |"
        )
        
    if rows and rows[0].get("ood_frac") and rows[0]["ood_frac"][0] is not None:
        print(f"\n*(fração OOD do split: {rows[0]['ood_frac'][0]*100:.1f}% -> teto de Acc(ID+OOD) = "
              f"{(1 - rows[0]['ood_frac'][0])*100:.1f}%, mesmo com classificador perfeito nas classes ID)*\n")


def save_csv(all_rows: list, path: str):
    """Salva os dados agregados em CSV."""
    import csv
    
    # Achatar as tuplas (mean, std) para colunas separadas
    flat_rows = []
    for r in all_rows:
        flat = {"corpus": r.get("corpus", ""), "kshot": r["kshot"], "label": r["label"], "folder": r["folder"], "folds": r["folds"]}
        for k in ["acc", "id_acc", "ood_frac", "f1", "aurc", "e_aurc", "auroc", "cov1", "cov5", "cov10",
                  "fpr95", "ece", "sgr_acc", "sgr_abs", "sgr_idcov", "sgr_oodrej"]:
            if r.get(k) and r[k][0] is not None:
                flat[f"{k}_mean"] = r[k][0]
                flat[f"{k}_std"] = r[k][1]
        flat_rows.append(flat)
        
    if not flat_rows:
        return
        
    fieldnames = list(flat_rows[0].keys())
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    print(f"\nCSV salvo em: {path}")


def plot_dataset(rows, kshot: str, corpus: str, lang: str, category: str):
    """Gera um Barplot comparativo de AURC e SGR@10%."""
    if not HAS_PLOT_LIBS:
        print("[!] Bibliotecas matplotlib/seaborn não instaladas. Gráfico pulado.")
        return
        
    # Pegar os 8 melhores métodos baseados na ordenação atual
    top_rows = rows[:8]
    if not top_rows:
        return
        
    labels = [r["label"] for r in top_rows]
    aurc_means = [r["aurc"][0] if r.get("aurc") and r["aurc"][0] is not None else 0 for r in top_rows]
    aurc_stds = [r["aurc"][1] if r.get("aurc") and r["aurc"][1] is not None else 0 for r in top_rows]
    
    sgr10_means = [r["cov10"][0] if r.get("cov10") and r["cov10"][0] is not None else 0 for r in top_rows]
    sgr10_stds = [r["cov10"][1] if r.get("cov10") and r["cov10"][1] is not None else 0 for r in top_rows]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(labels))
    width = 0.35
    
    color_aurc = 'tab:red'
    ax1.set_xlabel('Métodos')
    ax1.set_ylabel('AURC (↓ Menor é Melhor)', color=color_aurc)
    rects1 = ax1.bar(x - width/2, aurc_means, width, yerr=aurc_stds, label='AURC', color=color_aurc, capsize=5, alpha=0.7)
    ax1.tick_params(axis='y', labelcolor=color_aurc)
    
    ax2 = ax1.twinx()
    color_sgr = 'tab:blue'
    ax2.set_ylabel('SGR Coverage @ 10% (↑ Maior é Melhor)', color=color_sgr)
    rects2 = ax2.bar(x + width/2, sgr10_means, width, yerr=sgr10_stds, label='SGR@10%', color=color_sgr, capsize=5, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor=color_sgr)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    title = f"Avaliação de Risco Seletivo\nDataset: {corpus} | Categoria: {category} | Idioma: {lang} | Shot: {kshot.upper()}"
    plt.title(title)
    
    fig.tight_layout()
    
    save_path = os.path.join(REPORTS_DIR, f"{corpus}_{kshot}_chart.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Gráfico geral gerado e salvo em: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def get_parser():
    p = argparse.ArgumentParser(
        description="Comparação de métricas com agregação por Folds.",
    )
    p.add_argument("--mode",   choices=["full", "fast", "markdown"], default="full",
                   help="Modo de exibição: 'full' (todas as métricas), 'fast' ou 'markdown'. Padrão: full")
    p.add_argument("--kshot",  type=str, default=None,
                   help="Filtrar um kshot específico (ex: 1, 5, 10). Padrão: todos")
    p.add_argument("--corpus", type=str, default=None,
                   help="Nome do corpus (ex: IntentPTCorpus)")
    p.add_argument("--all",    action="store_true",
                   help="Rodar para todos os corpora que possuem 5 folds completos em todos os métodos.")
    p.add_argument("--sort",   type=str, default="cov10",
                   choices=list(SORT_KEYS.keys()),
                   help="Métrica de ordenação. Padrão: cov10")
    p.add_argument("--asc",    action="store_true",
                   help="Ordenação crescente (padrão: decrescente)")
    p.add_argument("--save",   type=str, default=None,
                   help="Salvar resultado em CSV (ex: outputs/comparison.csv)")
    p.add_argument("--plot",   action="store_true",
                   help="Gera um gráfico consolidado do dataset.")
    p.add_argument("--base-dir", type=str, default=None,
                   help="Sobrescreve AMBOS os diretórios base (baseline e laqda) para este caminho.")
    p.add_argument("--baseline-dir", type=str, default=None,
                   help=f"Diretório base do baseline. Padrão: {BASE_DIR_BASELINE}")
    p.add_argument("--laqda-dir", type=str, default=None,
                   help=f"Diretório base do laqda. Padrão: {BASE_DIR_LAQDA} (resultados pós-fix do "
                        "bug de gradient checkpointing).")
    return p


def check_kshot_complete(base_dir_map: dict, corpus: str, kshot: str):
    lang, _ = resolve_dataset_metadata(corpus)
    lang_dir = "br" if lang == "PT" else "en"
    for label, folder, suffix in METHODS:
        d = load_report_aggregated(base_dir_map[folder], lang_dir, folder, corpus, kshot, suffix)
        if d is None or d["folds_found"] < 5:
            return False
    return True

def get_all_corpora(base_dir_map: dict):
    # Une os corpora encontrados em CADA árvore (baseline pode ter um corpus que
    # o laqda ainda não tem em final_eval_fixed/, e vice-versa durante a
    # transição pro fix).
    corpora = set()
    for base_path in set(base_dir_map.values()):
        build_fs_cache(base_path)
        base_abs = os.path.abspath(base_path)
        for path in FS_CACHE.get(base_path, ()):
            rel = os.path.relpath(path, base_abs)
            parts = rel.split(os.sep)
            if len(parts) >= 3:
                corpora.add(parts[2])
    return sorted(corpora)

def main():
    args = get_parser().parse_args()
    if args.base_dir:
        base_dir_map = {"baseline": args.base_dir, "laqda": args.base_dir}
    else:
        base_dir_map = {
            "baseline": args.baseline_dir or BASE_DIR_BASELINE,
            "laqda": args.laqda_dir or BASE_DIR_LAQDA,
        }
    kshots    = [f"kshot_{args.kshot}"] if args.kshot else KSHOTS

    if args.all:
        all_corpora = get_all_corpora(base_dir_map)
        corpora_to_run = all_corpora
    else:
        if not args.corpus:
            print("Erro: A menos que --all seja usado, --corpus é obrigatório.")
            sys.exit(1)
        corpora_to_run = [args.corpus]

    all_rows_flat = []
    found_any = False

    for corpus in corpora_to_run:
        lang, category = resolve_dataset_metadata(corpus)
        lang_dir = "br" if lang == "PT" else "en"

        header_printed = False

        for kshot in kshots:
            if args.all and not check_kshot_complete(base_dir_map, corpus, kshot):
                continue

            rows = collect_rows_aggregated(base_dir_map, lang_dir, corpus, kshot, args.sort, args.asc)
            if not rows:
                continue

            if not header_printed:
                if args.mode == "markdown":
                    print(f"\n## DATASET: {corpus} | IDIOMA: {lang} | TIPO: {category}")
                    print(f"**Modo**: {args.mode} | **Ordenação**: {args.sort} ({'↑' if args.asc else '↓'})\n")
                else:
                    print(f"\n{'=' * 120}")
                    print(f"  DATASET: {corpus} | IDIOMA: {lang} | TIPO: {category}")
                    print(f"  modo={args.mode} | ordenação={args.sort} ({'↑' if args.asc else '↓'})")
                    print(f"{'=' * 120}")
                header_printed = True

            found_any = True
            if args.mode == "full":
                print_full(rows, kshot)
            elif args.mode == "markdown":
                print_markdown(rows, kshot)
            else:
                print_fast(rows, kshot)

            if args.plot:
                plot_dataset(rows, kshot, corpus, lang, category)

            for r in rows:
                r["kshot"] = kshot
                r["corpus"] = corpus
            all_rows_flat.extend(rows)

    if args.all and not found_any:
        print("Nenhum corpus/kshot possui 5 folds completos em TODOS os métodos.")
        return

    print(f"\n{'=' * 120}")
    print("  ★ = LAQDA   ◆ = LAQDA+SGR   (sem símbolo) = Baseline")
    print("  Valores representam Média ± Desvio Padrão considerando os Folds encontrados.")
    print(f"{'=' * 120}\n")

    if args.save:
        save_csv(all_rows_flat, args.save)


if __name__ == "__main__":
    main()
