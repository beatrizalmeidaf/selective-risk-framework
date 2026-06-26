import os
import zipfile
from collections import Counter

class MajorityVotingEnsemble:
    """
    Agrega as predições de múltiplos folds/modelos usando votação majoritária.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def vote(self, all_models_preds: list) -> list:
        if not all_models_preds:
            return []
            
        final_preds = []
        num_samples = len(all_models_preds[0])
        
        for i in range(num_samples):
            try:
                votes = [model_preds[i] for model_preds in all_models_preds]
                winner = Counter(votes).most_common(1)[0][0]
                final_preds.append(winner)
            except IndexError:
                print(f"Index error at line {i}. Ensure all prediction sets have the same length.")
                return []
                
        return final_preds

    def save(self, preds: list, filename: str = "prediction"):
        os.makedirs(self.output_dir, exist_ok=True)
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, "w", newline='\n', encoding='utf-8') as f:
            for p in preds:
                f.write(f"{p}\n")
                
        zip_path = os.path.join(self.output_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(output_path, arcname=filename)
            
        print(f"Ensemble saved successfully to {zip_path}")
