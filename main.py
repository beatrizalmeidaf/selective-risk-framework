import os
import subprocess

def run_preprocessing():
    print("==========================================")
    print("Iniciando Pré-Processamento do Framework")
    print("==========================================")
    
    script_path = os.path.join("scripts", "generate_ood_splits.py")
    if not os.path.exists(script_path):
        print(f"ERRO: Script {script_path} não encontrado.")
        return False
        
    print(f"--> Executando {script_path} para gerar/atualizar splits OOD...")
    result = subprocess.run(["python", script_path])
    
    if result.returncode == 0:
        print("--> Pré-processamento concluído com sucesso!")
        return True
    else:
        print("--> ERRO durante o pré-processamento.")
        return False

def main():
    print("Bem-vindo ao Selective Risk Framework!")
    if run_preprocessing():
        print("\nPara iniciar os testes, utilize o script 'bash scripts/run_test.sh'")
        print("Para treinar todos os modelos, utilize 'bash scripts/run_all.sh'")

if __name__ == "__main__":
    main()
