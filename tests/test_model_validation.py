import os
import tempfile
import json
import torch
import unittest
from methods.baselines.models.standard_classifier import BaselineClassifier
from methods.laqda.modules.laqda_module import LaqdaModule
from data.datamodule import StandardDataModule

class TestModelValidation(unittest.TestCase):

    def test_baseline_classifier_dimensions_and_freezing(self):
        print("\nTestando BaselineClassifier (Dimensões e Congelamento)")
        num_classes = 5
        num_freeze = 6
        
        # Usar o modelo padrão do projeto que já está em cache
        model = BaselineClassifier("bert-base-uncased", num_classes=num_classes, num_freeze=num_freeze)
        
        # Testar Forward
        texts = ["Olá mundo", "Testando o selective risk framework."]
        features, logits = model(texts)
        
        # Dimensões esperadas (bert-base hidden size é 768)
        self.assertEqual(features.shape, (2, 768))
        self.assertEqual(logits.shape, (2, num_classes))
        print("Dimensões do BaselineClassifier: OK")
        
        # Verificar se as 6 primeiras camadas e os embeddings estão congelados
        # bert-base-uncased tem 12 camadas. As primeiras 6 (0 a 5) devem estar congeladas.
        for name, param in model.encoder.named_parameters():
            if any(f"layer.{i}." in name for i in range(num_freeze)) or "embeddings" in name:
                self.assertFalse(param.requires_grad, f"Parâmetro {name} não deveria ter gradientes!")
                
        # Verificar se as camadas restantes (6 a 11) e o classificador estão ativos
        for name, param in model.encoder.named_parameters():
            if any(f"layer.{i}." in name for i in range(num_freeze, 12)):
                self.assertTrue(param.requires_grad, f"Camada {name} deveria estar destravada!")
        
        print("Congelamento de Camadas (BaselineClassifier): OK")

    def test_laqda_encoder_freezing(self):
        print("\nTestando LaqdaModule (Congelamento de Camadas) ---")
        model = LaqdaModule("bert-base-uncased", nway=3, kshot=2, qshot=2, num_freeze=6)
        
        # Para num_freeze=6, a camada 0-5 e embeddings devem estar congelados
        for name, param in model.encoder.model.named_parameters():
            if any(f"layer.{i}." in name for i in range(6)) or "embeddings" in name:
                self.assertFalse(param.requires_grad, f"LaqdaEncoder {name} deveria estar congelado!")
            if any(f"layer.{i}." in name for i in range(6, 12)):
                self.assertTrue(param.requires_grad, f"LaqdaEncoder {name} deveria estar ativo!")
        print("Congelamento de Camadas (LaqdaModule): OK")

    def test_datamodule_data_leakage_prevention(self):
        print("\nTestando Data Leakage no StandardDataModule")
        
        # Criar dados temporários para simular um dataset com classes ID e OOD
        with tempfile.TemporaryDirectory() as tmpdir:
            fold_dir = os.path.join(tmpdir, "01")
            os.makedirs(fold_dir)
            
            # Treino tem apenas classes ID (A e B)
            train_data = [
                {"sentence": "frase de treino 1", "class_name": "A"},
                {"sentence": "frase de treino 2", "class_name": "B"},
            ]
            # Validação tem classes ID e OOD (C)
            valid_data = [
                {"sentence": "frase de validação 1", "class_name": "A"},
                {"sentence": "frase de validação 2", "class_name": "C"}, # OOD class
            ]
            # Teste tem classes ID e OOD (C)
            test_data = [
                {"sentence": "frase de teste 1", "class_name": "B"},
                {"sentence": "frase de teste 2", "class_name": "C"}, # OOD class
            ]
            
            with open(os.path.join(fold_dir, "train.jsonl"), "w") as f:
                for item in train_data: f.write(json.dumps(item) + "\n")
            with open(os.path.join(fold_dir, "valid.jsonl"), "w") as f:
                for item in valid_data: f.write(json.dumps(item) + "\n")
            with open(os.path.join(fold_dir, "test.jsonl"), "w") as f:
                for item in test_data: f.write(json.dumps(item) + "\n")
                
            # Configurar o DataModule
            # keep_unknown_classes=False na validação deve descartar a classe OOD "C"
            # keep_unknown_classes=True no teste deve manter a classe OOD "C" como class_id = -1
            datamodule = StandardDataModule(tmpdir, fold="01", keep_unknown_classes=True)
            datamodule.setup()
            
            # Verificar mapeamento de classes (apenas A e B do treino são ID)
            self.assertIn("A", datamodule.labels_dict)
            self.assertIn("B", datamodule.labels_dict)
            self.assertNotIn("C", datamodule.labels_dict)
            
            # 1. Validação NÃO PODE conter a classe OOD (classe C deve ser descartada totalmente)
            val_loader = datamodule.val_dataloader()
            for batch in val_loader:
                # O batch é retornado como dicionário pelo JSONLDataset
                labels = batch["class_id"]
                # Não pode existir class_id == -1 na validação!
                self.assertTrue((labels != -1).all(), "Data Leakage! A classe OOD não foi descartada na validação.")
                
            # 2. Teste DEVE manter a classe OOD rotulada como -1
            test_loader = datamodule.test_dataloader()
            has_ood = False
            for batch in test_loader:
                labels = batch["class_id"]
                if (labels == -1).any():
                    has_ood = True
            self.assertTrue(has_ood, "Erro: A classe OOD deveria ter sido rotulada como -1 no Test Set.")
            
            print("Prevenção de Data Leakage no DataModule: OK")

if __name__ == "__main__":
    unittest.main()
