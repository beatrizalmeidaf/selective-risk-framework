# Selective Risk Framework

Esse repositório contém a implementação do framework de avaliação e mitigação de risco em classificação seletiva e detecção de dados fora de distribuição (OOD - Out-of-Distribution). O framework integra o método **LAQDA** (Label-Aware Quantitative Data Analysis), algoritmos de **Baselines** (MSP, Energy Score, Distância de Mahalanobis, kNN) e o controlador de rejeição controlada **SGR** (Selection with Guaranteed Risk).

---

## Página do Projeto (GitHub Pages)

A página pública do projeto — abstract, pipeline, gráficos interativos e todas as tabelas de resultados do artigo ACL — fica em [`site/`](site/).

*   **Pré-visualizar localmente**:
    ```bash
    python -m http.server 8000 --directory site
    # abre http://localhost:8000
    ```
*   **Publicar**: o workflow [`.github/workflows/pages.yml`](.github/workflows/pages.yml) publica a pasta `site/` a cada push na `main`. É necessário configurar uma única vez em **Settings → Pages → Source: GitHub Actions**. A URL final é `https://beatrizalmeidaf.github.io/selective-risk-framework/`.

Os números da página são transcritos de `docs/conferences/ACL/acl_latex.tex`; ao atualizar o artigo, atualize também os blocos de dados `AURC` e `COV` em `site/index.html`.

---

## Estrutura do Framework

A estrutura do projeto está organizada de forma modular:

*   **`configs/`**: Arquivos de configuração centralizados do modelo (ex: `model_config.yaml`).
*   **`data/`**: Componentes de dados, datamodules do PyTorch Lightning/customizados e samplers episódicos de k-shot.
*   **`methods/`**:
    *   **`laqda/`**: Módulos e executáveis do LAQDA (Label-Aware Encoder, QDA Sampler, Loss contrastiva, cli de treino/inferência).
    *   **`baselines/`**: Classificadores padrão, scorers probabilísticos (Maximum Softmax Probability - MSP, Energy Score), scorers de distância (Distância de Mahalanobis, kNN dentro de `distance/`), e scorers baseados em técnicas SOTA (**GradNorm** analítico, **ReAct** com truncamento de ativações anômalas, e **ConjNorm** por normalização de cosseno na hiperesfera dentro de `sota/`).
    *   **`sgr/`**: O algoritmo **SGR** (Selection with Guaranteed Risk) para controle estatístico de risco e limiarização pós-hoc.
    *   **`metrics/`**: A suíte unificada de avaliação (Acurácia, F1-Score, ECE calibrado, AUROC, FPR@95, AUPR e AURC).
*   **`tests/`**: Testes automatizados para validação matemática e de integridade dos componentes do framework.

---

## Configuração do Ambiente

O framework suporta duas formas principais de execução: nativa via `uv` (gerenciador rápido de pacotes Python) ou containerizada via `Docker`.

### Opção A: Execução Nativa (Recomendado para Cluster/DGX)

O projeto utiliza o `uv` para automatizar e acelerar a gerência de pacotes virtuais.

1.  **Instalar o UV e Sincronizar Dependências**:
    Na raiz do projeto, inicialize o ambiente virtual integrado:
    ```bash
    make setup-env
    ```
    *(Isso instalará a ferramenta `uv` e gerará automaticamente uma pasta oculta `.venv` com todas as dependências especificadas instaladas).*

2.  **Ativar o Ambiente Virtual**:
    Antes de rodar qualquer script do framework, ative o ambiente virtual criado:
    ```bash
    source .venv/bin/activate
    ```

### Opção B: Execução Containerizada (Docker)

Se você preferir executar o treinamento de forma isolada usando contêineres com suporte a GPU:

1.  **Construir a Imagem**:
    ```bash
    make laqda-install
    ```
    *(Roda internamente o comando `docker compose build laqda`).*

2.  **Subir o Container com GPU**:
    ```bash
    docker compose up laqda
    ```

---

## Como Executar

### 1. Pré-processamento e Divisões OOD

Antes de executar qualquer treinamento, garanta a geração das divisões In-Distribution (ID) e Out-of-Distribution (OOD) determinísticas para todos os folds. O framework possui um ponto de entrada centralizado para isso:

```bash
python main.py
```
*(Esse comando garante que o arquivo `configs/ood_splits.json` seja devidamente preenchido mapeando todos os corpus presentes nas pastas `datasets-br-nlp` e `datasets-en-nlp`).*

### 2. Treinamento do LAQDA

O LAQDA pode ser treinado com ou sem a ativação da otimização do threshold SGR. 

*   **LAQDA Tradicional (Sem SGR)**:
    Opera como um classificador fechado padrão com **100% de cobertura**. Se receber dados OOD, ele tentará encaixar em uma classe conhecida (devido à obrigatoriedade de chute), podendo gerar alta taxa de Falsos Positivos.
    ```bash
    python -m methods.laqda.cli.train \
        --dataset_dir data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot \
        --fold 01 \
        --save_dir outputs/laqda/IntentPTCorpus/fold_01
    ```
*   **LAQDA Seletivo (Com SGR)**:
    Acopla uma camada de segurança estatística rigorosa. Estima e salva um limiar crítico ($\theta^*$) correspondente a **5% de risco**. Em teste, se a confiança for menor que $\theta^*$, o modelo **se recusa a prever (abstenção)**. Isso derruba a cobertura geral, mas garante matematicamente que as respostas dadas pelo modelo tenham menos de 5% de erro, estancando os Falsos Positivos.
    ```bash
    python -m methods.laqda.cli.train \
        --dataset_dir data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot \
        --fold 01 \
        --save_dir outputs/laqda_sgr/IntentPTCorpus/fold_01 \
        --use_sgr
    ```

### 3. Treinamento de Baselines

Para rodar os baselines, treine o classificador supervisionado base. Ele carregará os parâmetros de OOD de `configs/methods_config.yaml` e executará e salvará automaticamente o benchmark individual de todas as técnicas (MSP, Energy Score, Distância de Mahalanobis, kNN, bem como as abordagens SOTA: GradNorm, ReAct e ConjNorm) no conjunto de testes:

```bash
python -m methods.baselines.cli.train_baseline \
    --dataset_dir data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot \
    --fold 01 \
    --save_dir outputs/baseline/IntentPTCorpus/fold_01
```

### 4. Suporte a Treinamento K-shot (Poucos Exemplos)

Todos os scripts de treinamento (`train_baseline.py` e `train.py`) e de inferência (`infer.py`) aceitam o argumento opcional `--kshot <K>` (ex: `--kshot 5`). 

Quando especificado:
1. O `StandardDataModule` realiza uma amostragem determinística para limitar o conjunto de treino a no máximo `K` exemplos por classe.
2. Os modelos e tensores gerados são salvos em subdiretórios estruturados como `kshot_<K>/` (ex: `outputs/baseline/IntentPTCorpus/fold_01/kshot_5/`).

### 5. Executando Inferências e Avaliação (LAQDA)

Para rodar inferências no conjunto de testes com o modelo LAQDA treinado:

```bash
python -m methods.laqda.cli.infer \
    --dataset_dir data/datasets/datasets-br-nlp/intent/IntentPTCorpus/few_shot \
    --fold 01 \
    --model_paths outputs/laqda/IntentPTCorpus/fold_01/kshot_5/acc_best_model.pth \
    --output_dir outputs/laqda/IntentPTCorpus/fold_01 \
    --kshot 5
```

### 6. Consolidando Métricas (LAQDA)

Após salvar os arquivos correspondentes a cada execução de inferência, você pode gerar o relatório pivotado das métricas:

```bash
make laqda-eval
```

---

## Testes Automatizados

O framework possui uma bateria de testes unitários para validar se os scorers OOD e de distância estão funcionando corretamente e gerando saídas nos formatos corretos.

Para executar os testes:
```bash
make test-baselines
```
*(Ou execute diretamente pelo interpretador: `python -m tests.test_metrics`).*

---

## Execução e Monitoramento via Cluster (SLURM)

O framework possui suporte para execução e monitoramento de experimentos no cluster usando SLURM de forma assíncrona.

### 1. Execução Completa do Benchmark

Para submeter todos os algoritmos (Baselines, KNN-Contrastive, LAQDA e LAQDA+SGR) por 100 épocas sobre todos os folds (`fold_01` a `fold_05`) e múltiplos k-shots (`1`, `5`, `10`), utilize os scripts abaixo de acordo com o idioma desejado:

*   **Para Datasets em Português (PT)**:
    ```bash
    bash scripts/run_all_pt.sh
    ```
*   **Para Datasets em Inglês (EN)**:
    ```bash
    bash scripts/run_all_en.sh
    ```

Esses scripts disparam automaticamente múltiplos jobs paralelos no cluster usando a partição `h100n3` e alocando uma GPU H100 por execução. Os logs de saída serão salvos em `outputs/logs_slurm/pt/` e `outputs/logs_slurm/en/`.

### 2. Acompanhamento dos Jobs

*   **Verificar Fila de Jobs**:
    ```bash
    squeue -u $USER
    ```
*   **Acompanhar Saída de um Job em Tempo Real**:
    ```bash
    tail -f outputs/logs_slurm/pt/base_<CORPUS>_<FOLD>_<K>_<JOB_ID>.log
    ```
*   **Cancelar Todos os seus Jobs**:
    ```bash
    scancel -u $USER
    ```

---

## Consolidação de Resultados e Gráficos

O script `scripts/compare.py` é responsável por varrer os logs consolidados em `outputs/final_eval/` e gerar relatórios agregados e análises visuais.

### 1. Agregação Cross-Fold (Média ± Desvio Padrão)

O script detecta automaticamente a linguagem e categoria do dataset a partir da estrutura em `data/datasets/` e calcula a média e desvio padrão entre todos os 5 folds executados.

*   **Gerar Tabela de Comparação (Exemplo)**:
    ```bash
    python scripts/compare.py --corpus IntentPTCorpus
    ```

### 2. Geração de Relatório Visual

Ao passar a flag `--plot`, o script gera gráficos comparativos cruzando a métrica base **AURC** e a cobertura de garantia de risco **SGR@10%**, salvando o gráfico `.png` em `outputs/final_eval/reports/`.

*   **Gerar Tabela + Gráfico Consolidado**:
    ```bash
    python scripts/compare.py --corpus IntentPTCorpus --plot
    ```

---

## Comandos do Makefile

O `Makefile` atua como o orquestrador de atalhos rápidos do framework:

| Comando | Descrição |
|:---|:---|
| `make setup-env` | Cria a `.venv` local e instala dependências via `uv`. |
| `make laqda-install` | Cria a imagem Docker para o LAQDA. |
| `make laqda-train` | Exibe a ajuda com parâmetros do script de treino do LAQDA. |
| `make laqda-infer` | Exibe exemplo de parâmetros do script de inferência do LAQDA. |
| `make train-baseline` | Exibe exemplo de treino do modelo baseline. |
| `make test-baselines` | Executa a suíte de testes unitários (`tests/test_metrics.py`). |
| `make run-pipeline` | Dispara o pipeline de benchmark completo (PT e EN) no SLURM. |
| `make laqda-eval` | Consolida e apresenta os relatórios e métricas consolidadas. |
| `make laqda-lint` | Valida sintaxe e compilação dos módulos Python. |
| `make laqda-clean` | Remove arquivos temporários de cache e pastas `__pycache__`. |

---

## Matriz de Métricas Implementada

Todas as métricas especificadas abaixo são calculadas e exportadas automaticamente no fim de cada ciclo de teste:

| Eixo de Avaliação | Métrica | Definição / Objetivo |
|:---|:---|:---|
| **Classificação Base** | Acurácia, Balanced Acc, F1-Score | Mede o desempenho sobre as previsões aceitas pelo modelo. (Balanced Acc foca no equilíbrio de aprendizado interclasses). |
| **Classificação Base** | Precision / Recall (Macro) | Precision foca na pureza dos acertos (menos falsos positivos), Recall foca em não esquecer as instâncias de cada classe. |
| **Calibração** | ECE (Expected Calibration Error) | Quantifica a discrepância entre a confiança do modelo e a sua precisão empírica real. |
| **Separação OOD** | AUROC / FPR@95 / FPR@90 | Avalia a separabilidade entre dados In-Distribution (ID) vs Out-of-Distribution (OOD). FPR mede os penetras que passam quando travamos o recall em 90/95%. |
| **Separação OOD** | TNR@95 / AUPR-IN / AUPR-OUT | AUPR mede a separabilidade sob desbalanceamento severo de classes, e TNR mostra a "Taxa de Verdadeiros Negativos" (rejeição de OOD). |
| **Predição Seletiva** | Curva RC e AURC / E-AURC | Avalia o trade-off entre risco e cobertura. O E-AURC subtrai o risco base do modelo para normalizar comparações. |
| **Predição Seletiva** | Risk @ Cobertura Fixo | Mede o risco quando a cobertura do sistema é travada cirurgicamente em patamares (ex: 50%, 80%, 95%). |
| **Garantia de Risco** | Cobertura do SGR | Mensura a cobertura empírica alcançada ao fixar níveis teóricos restritos de risco (via PAC Bounds). |
| **Abstenção Ativa** | Taxa Abstenção / Acc Aceita | Exclusiva do LAQDA com SGR. Mede a porcentagem real de testes classificados como OOD/Rejeição (-1) e a acurácia do que sobrou na peneira. |classificados como OOD/Rejeição (-1) e a acurácia do que sobrou na peneira. |


./scripts/run_all_pt.sh
./scripts/run_all_en.sh

python scripts/compare.py --corpus IntentPTCorpus --plot

