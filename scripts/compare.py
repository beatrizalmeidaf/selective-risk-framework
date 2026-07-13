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
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs", "final_eval")

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

def resolve_dataset_metadata(corpus: str):
    """Procura o dataset nas pastas data/datasets para inferir Idioma e Categoria."""
    base_data = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "datasets")
    search_pattern = os.path.join(base_data, "datasets-*-nlp", "*", corpus)
    matches = glob.glob(search_pattern)
    
    if not matches:
        return "Unknown", "Unknown"
        
    match_path = matches[0]
    parts = match_path.split(os.sep)
    
    # Ex: .../datasets-br-nlp/intent/IntentPTCorpus
    dataset_group = parts[-3]
    lang = "PT" if "br" in dataset_group.lower() else "EN"
    category = parts[-2].capitalize()
    return lang, category


def load_report_aggregated(base_path: str, lang_dir: str, folder: str, corpus: str, kshot: str, suffix: str):
    """Carrega os resultados de múltiplos folds (01 a 05) e calcula a média e o desvio padrão."""
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
        path = os.path.join(base_path, lang_dir, folder, corpus, fold, kshot, fname)
        
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
                for k in metrics.keys():
                    if k in d and d[k] is not None:
                        metrics[k].append(d[k])
            folds_found += 1
            
    if folds_found == 0:
        return None
        
    aggregated = {"folds_found": folds_found}
    for k, vlist in metrics.items():
        if len(vlist) > 0:
            aggregated[k] = (np.mean(vlist), np.std(vlist))
        else:
            aggregated[k] = (None, None)
            
    return aggregated


def collect_rows_aggregated(base_path: str, lang_dir: str, corpus: str, kshot: str, sort_key: str, asc: bool):
    """Coleta e ordena as linhas agregadas para todos os métodos."""
    metric_field = SORT_KEYS.get(sort_key, "sgr_coverage_at_risk_10")
    rows = []
    
    for label, folder, suffix in METHODS:
        d = load_report_aggregated(base_path, lang_dir, folder, corpus, kshot, suffix)
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


def save_csv(all_rows: list, path: str):
    """Salva os dados agregados em CSV."""
    import csv
    
    # Achatar as tuplas (mean, std) para colunas separadas
    flat_rows = []
    for r in all_rows:
        flat = {"kshot": r["kshot"], "label": r["label"], "folder": r["folder"], "folds": r["folds"]}
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
    
    save_path = os.path.join(BASE_DIR, "reports", f"{corpus}_{kshot}_chart.png")
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
    p.add_argument("--mode",   choices=["full", "fast"], default="full",
                   help="Modo de exibição: 'full' (todas as métricas) ou 'fast'. Padrão: full")
    p.add_argument("--kshot",  type=str, default=None,
                   help="Filtrar um kshot específico (ex: 1, 5, 10). Padrão: todos")
    p.add_argument("--corpus", type=str, required=True,
                   help="Nome do corpus obrigatorio (ex: IntentPTCorpus)")
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
                   help="Caminho base alternativo para outputs/final_eval")
    return p


def main():
    args = get_parser().parse_args()
    base_path = args.base_dir if args.base_dir else BASE_DIR
    corpus    = args.corpus
    kshots    = [f"kshot_{args.kshot}"] if args.kshot else KSHOTS
    
    lang, category = resolve_dataset_metadata(corpus)
    lang_dir = "br" if lang == "PT" else "en"

    print(f"\n{'=' * 120}")
    print(f"  DATASET: {corpus} | IDIOMA: {lang} | TIPO: {category}")
    print(f"  modo={args.mode} | ordenação={args.sort} ({'↑' if args.asc else '↓'})")
    print(f"{'=' * 120}")

    all_rows_flat = []
    for kshot in kshots:
        rows = collect_rows_aggregated(base_path, lang_dir, corpus, kshot, args.sort, args.asc)
        if not rows:
            print(f"\n  [!] Nenhum resultado consolidado encontrado para {kshot}.")
            continue

        if args.mode == "full":
            print_full(rows, kshot)
        else:
            print_fast(rows, kshot)

        if args.plot:
            plot_dataset(rows, kshot, corpus, lang, category)

        for r in rows:
            r["kshot"] = kshot
        all_rows_flat.extend(rows)

    print(f"\n{'=' * 120}")
    print("  ★ = LAQDA   ◆ = LAQDA+SGR   (sem símbolo) = Baseline")
    print("  Valores representam Média ± Desvio Padrão considerando os Folds encontrados.")
    print(f"{'=' * 120}\n")

    if args.save:
        save_csv(all_rows_flat, args.save)


if __name__ == "__main__":
    main()
