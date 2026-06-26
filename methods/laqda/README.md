# LAQDA (Label-Aware Quantitative Data Analysis)

A implementação refatorada do método LAQDA integrada ao `selective-risk-framework`. 
O LAQDA utiliza um encoder Label-Aware que produz protótipos representativos para as classes e um sampler transdutivo (QDA) que seleciona exemplos representativos baseados nas consultas para melhorar o espaço latente de suporte e estimar risco.

## Estrutura do Módulo

- `configs/`: Definições hiperparamétricas centralizadas em YAML.
- `models/`: Implementação PyTorch do Label-Aware Encoder e do QDA Sampler.
- `losses/`: Função de Perda Contrastiva especializada.
- `datamodules/`: Sampler episódico de k-shot para LAQDA.
- `trainers/` e `evaluators/`: Classes orquestradoras dos ciclos do PyTorch.
- `inference/`: Lógica de predição com os folds de modelos e sistema de votação (Ensemble).
- `metrics/`: Consolidador de logs CSV.
- `cli/`: Entrypoints para o Makefile executar ações no terminal.

## Configuração do Ambiente e Execução Nativa (Sem Docker)

O projeto `selective-risk-framework` usa uma instalação unificada de ambiente via `uv`, que atende tanto o LAQDA quanto futuros modelos integrados. Se você não possuir privilégios de Docker (`sudo`) ou prefere rodar localmente no host (ex: nós DGX-H100), utilize este fluxo:

### 1. Criando o Ambiente Global (Apenas na Primeira Vez)
Na raiz do framework, inicie o projeto global usando o `uv`:
```bash
make setup-env
```
*(Isso gerará a pasta oculta `.venv` contendo o Python e instalará globalmente as dependências de todos os módulos listados no projeto).*

### 2. Ativando o Ambiente
Sempre antes de executar qualquer componente do framework, ative o ambiente global:
```bash
source .venv/bin/activate
```

### 3. Treinando o LAQDA
Com o ambiente ativado, execute o fluxo de treinamento, apontando seus datasets:
```bash
make laqda-train
# O comando make acima mostrará os argumentos. Um uso real seria:
# python -m methods.laqda.cli.train --train_file /caminho/treino.jsonl --config methods/laqda/configs/default.yaml
```

### 4. Avaliando / Inferindo (Ensemble)
Para submissões CodaBench ou inferência nos conjuntos de testes com múltiplos folds:
```bash
make laqda-test
# Um uso real seria:
# python -m methods.laqda.cli.infer --train_file /caminho/treino.jsonl --test_file /caminho/teste.jsonl --model_paths outputs/fold1.pth outputs/fold2.pth
```

### 5. Consolidando Métricas
```bash
make laqda-eval
```
*(Procura por arquivos `result.csv` gerados pelas inferências e pivota um relatório consolidado com médias).*
