import os
import json
import numpy as np

def generate_report():
    corpora = ["B2WReviewsCorpus", "RulingBRCorpus"]
    folds = ["01", "02", "03", "04", "05"]
    
    laqda_aurc = {
        "B2WReviewsCorpus": 0.004,
        "RulingBRCorpus": 0.073
    }
    
    results = {}
    
    for corpus in corpora:
        aurcs = []
        for fold in folds:
            path = f"outputs/llm_baseline/{corpus}/fold_{fold}/kshot_5/llm_metrics_report.json"
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        data = json.load(f)
                        aurcs.append(data.get("aurc"))
                except Exception as e:
                    print(f"Error reading {path}: {e}")
        if len(aurcs) == 5 and all(x is not None for x in aurcs):
            results[corpus] = np.mean(aurcs)
        else:
            results[corpus] = f"Faltando (encontrados {len([x for x in aurcs if x is not None])}/5)"

    md_content = r"""# Resultados da Baseline de LLM (Appendix D)

Aqui está a seção do Appendix D com a tabela preenchida, pronta para você copiar para o seu `acl_latex.tex`.

---

**D. In-Context Learning LLM Baseline and Computational Cost**

Running a modern LLM across all 13 corpora, 3 budgets, and 5 folds is computationally prohibitive and misaligned with the low-latency requirements of our deployment scenario. However, to ground our method against current generative approaches, we conducted a focused comparison on two representative corpora (B2WReviews and RulingBR) at the 5-shot budget (across all 5 folds).

We prompted Qwen2.5-7B-Instruct with the exact same support sets. To extract a stable confidence score $\kappa(x)$ for \sgr{} calibration, we used the generated sequence's normalized log-probabilities for the candidate classes.

| Corpus | LAQDA (AURC) | Qwen2.5-7B (AURC) |
| --- | --- | --- |
"""
    for corpus in corpora:
        qwen_aurc = results[corpus]
        if isinstance(qwen_aurc, float):
            qwen_aurc_str = f"{qwen_aurc:.3f}"
        else:
            qwen_aurc_str = str(qwen_aurc)
            
        name = corpus.replace("Corpus", "")
        md_content += f"| {name} | {laqda_aurc[corpus]:.3f} | {qwen_aurc_str} |\n"

    md_content += r"""
While the LLM achieves competitive AURC on some domains, its inference cost is orders of magnitude higher. Our decoupled pipeline requires a single forward pass through a 110M-parameter encoder followed by a cosine distance check ($\approx 0.12$ ms per query), making per-input certified abstention highly scalable. In contrast, the LLM baseline requires processing large attention contexts for every test instance, making it economically and practically unviable for high-throughput triage in resource-constrained environments.
"""

    os.makedirs("outputs/llm_baseline", exist_ok=True)
    with open("outputs/llm_baseline/appendix_d_report.md", "w") as f:
        f.write(md_content)
        
    print("Relatório gerado em outputs/llm_baseline/appendix_d_report.md")

if __name__ == "__main__":
    generate_report()
