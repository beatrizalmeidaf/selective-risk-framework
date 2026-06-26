import pandas as pd
import os
import glob

def collate_metrics(base_dir: str, output_path: str):
    """
    Agrega as métricas de diferentes datasets e folds em um único relatório consolidado.
    Refatorado a partir de utils/merge_results.py e all_results.py.
    """
    all_data_rows = []
    
    # Encontra todos os result.csv em qualquer subdiretório
    search_pattern = os.path.join(base_dir, "**", "result.csv")
    result_files = glob.glob(search_pattern, recursive=True)
    
    for result_file in result_files:
        try:
            df_temp = pd.read_csv(result_file, header=None)
            
            # Formato esperado gerado pelo laqda original:
            # commont, NOME_EXPERIMENTO, data, CAMINHO, shot, SHOT, acc, ACC, f1, F1
            for index, row in df_temp.iterrows():
                try:
                    all_data_rows.append({
                        "experiment": row[1],
                        "dataset": os.path.basename(os.path.dirname(row[3])), # aproximação do nome do dataset
                        "shot": pd.to_numeric(row[5], errors="coerce"),
                        "acc": pd.to_numeric(row[7], errors="coerce"),
                        "f1": pd.to_numeric(row[9], errors="coerce") if len(row) > 9 else None
                    })
                except IndexError:
                    continue
        except Exception as e:
            print(f"Erro ao ler {result_file}: {e}")
            
    if not all_data_rows:
        print("Nenhuma métrica encontrada.")
        return None
        
    final_df = pd.DataFrame(all_data_rows)
    
    # Gera um relatório dinâmico pivotando pelos shots
    pivot_df = final_df.pivot_table(index=["dataset", "experiment"], columns="shot", values="acc", aggfunc='mean').reset_index()
    pivot_df.columns.name = None
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pivot_df.to_csv(output_path, index=False)
    print(f"Relatório consolidado salvo em {output_path}")
    return pivot_df
