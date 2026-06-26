import argparse
from ..metrics.collator import collate_metrics

def get_parser():
    parser = argparse.ArgumentParser(description="Agregador de métricas do LAQDA")
    parser.add_argument('--base_dir', type=str, required=True, help='Diretório base contendo subdiretórios com result.csv')
    parser.add_argument('--output_path', type=str, default='./laqda_consolidated_metrics.csv', help='Caminho do CSV de saída')
    return parser

def main():
    args = get_parser().parse_args()
    collate_metrics(args.base_dir, args.output_path)

if __name__ == "__main__":
    main()
