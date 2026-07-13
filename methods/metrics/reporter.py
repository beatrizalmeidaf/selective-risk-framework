import json
import os
import torch
import numpy as np
import math

from .classification import compute_accuracy_f1
from .calibration import compute_ece
from .selective_eval import evaluate_selective
from .ood_eval import evaluate_ood

class MetricsReporter:
    """
    Agrega todas as métricas solicitadas.
    """
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        
    def generate_report(self, confidences, preds, targets, id_scores=None, ood_scores=None, model=None, prefix="val", **extras):
        """
        Gera o relatório consolidado de métricas.
        - confidences: Tensor com as confianças (probabilidade da classe predita ou score de incerteza invertido).
        - preds: Tensor com as classes preditas.
        - targets: Tensor com as labels reais.
        - id_scores, ood_scores: Tensores para avaliação OOD separada (opcional durante o treino base).
        - extras: Métricas adicionais a serem incluídas no relatório.
        """
        report = {}
        
        # 1. Classificação Base
        report.update(compute_accuracy_f1(preds, targets))
        
        # 2. Calibração
        report.update(compute_ece(confidences, preds, targets))
        
        # 3. Predição Seletiva (Risco-Cobertura)
        report.update(evaluate_selective(confidences, preds, targets))
        
        # 4. Separação OOD (Se fornecido)
        if id_scores is not None and ood_scores is not None and len(ood_scores) > 0:
            report.update(evaluate_ood(id_scores, ood_scores))
            
        # 5. Eficiência Computacional
        if model is not None:
            from .efficiency import compute_parameters
            report.update(compute_parameters(model))
            
        # 6. Adicionar métricas extras injetadas (ex: Abstenção SGR)
        report.update(extras)
            
        # Converter numpy floats para python floats para JSON
        for k, v in report.items():
            if isinstance(v, (np.float32, np.float64, np.float16)):
                report[k] = float(v)
            if isinstance(report[k], float):
                if math.isinf(report[k]) or math.isnan(report[k]):
                    report[k] = str(report[k])
                
        # Exibir na tela
        print(f"\n[{prefix.upper()}] Metricas de Avaliação:")
        for k, v in report.items():
            print(f"  - {k}: {v:.4f}")
            
        # Salvar se diretório for providenciado
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, f"{prefix}_metrics_report.json")
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=4)
            print(f"Relatório salvo em {save_path}")
            
        return report
