.PHONY: laqda-install laqda-train laqda-test laqda-eval laqda-lint laqda-format laqda-clean

laqda-install:
	@echo "Construindo a imagem Docker do LAQDA..."
	@echo "Nota: Requer permissões de administrador (sudo) ou pertencer ao grupo 'docker'."
	docker compose build laqda

setup-env:
	@echo "Inicializando o ambiente virtual global do framework com UV..."
	pip install --user uv
	uv sync
	@echo "Ambiente global criado! Ative-o rodando: source .venv/bin/activate"

laqda-train:
	@echo "Executando treinamento LAQDA..."
	@echo "Lembre-se de ativar o ambiente virtual (ex: source .venv/bin/activate) antes de rodar este comando."
	@echo "Exemplo: python -m methods.laqda.cli.train --train_file <path>"
	python -m methods.laqda.cli.train --help

laqda-infer:
	@echo "Executando inferência LAQDA..."
	@echo "Lembre-se de ativar seu ambiente virtual (ex: source .venv/bin/activate) antes de rodar este comando."
	@echo "Exemplo: python -m methods.laqda.cli.infer --train_file <path> --test_file <path> --model_path <path>"

train-baseline:
	@echo "Treinando Classificador Baseline Genérico..."
	@echo "Exemplo: python -m methods.baselines.cli.train_baseline --train_file <path> --valid_file <path>"

test-baselines:
	@echo "Testando algoritmos de Baselines OOD..."
	python -m tests.test_metrics

run-pipeline:
	@echo "Executando o pipeline unificado (LAQDA + Teste de Baselines)..."
	bash run_all.sh

laqda-eval:
	@echo "Consolidando métricas do LAQDA..."
	python -m methods.laqda.cli.collate --help

laqda-lint:
	@echo "Verificando sintaxe e imports do LAQDA..."
	python -m py_compile data/datamodule.py
	python -m py_compile data/datasets/*.py
	python -m py_compile methods/laqda/models/*.py
	python -m py_compile methods/laqda/losses/*.py
	python -m py_compile methods/laqda/trainers/*.py
	python -m py_compile methods/laqda/cli/*.py
	@echo "Linting finalizado com sucesso."

laqda-format:
	@echo "Formatador não configurado. Sugestão: instalar black ou ruff."

laqda-clean:
	@echo "Limpando arquivos temporários e cache..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	@echo "Limpeza concluída."
