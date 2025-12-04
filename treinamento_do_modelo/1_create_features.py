import pandas as pd
import os
import sys

# Adiciona o diretório 'aplicacao' ao caminho do Python para podermos importar nossos módulos traduzidos
# O '..' significa "voltar uma pasta" (sair de treinamento_do_modelo e ir para a raiz)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'aplicacao'))

from data_loader import CarregadorDados  # Importando a classe traduzida
from feature_extractor import ExtratorMetaCaracteristicas  # Importando a classe traduzida

def gerar_features_treinamento():
    # Caminhos dos diretórios (ajustados automaticamente baseado na localização deste script)
    caminho_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    caminho_dados = os.path.join(caminho_base, 'dados_para_treinamento', 'bancos_brutos')
    caminho_gabarito = os.path.join(caminho_base, 'dados_para_treinamento', 'gabarito_master.csv')
    caminho_saida = os.path.join(caminho_base, 'dados_para_treinamento', 'training_features.csv')

    # Inicializa nossas ferramentas
    carregador = CarregadorDados()
    extrator = ExtratorMetaCaracteristicas()

    # Carrega o gabarito (a "cola" que diz qual é o tipo correto de cada coluna)
    if not os.path.exists(caminho_gabarito):
        print("Erro: Arquivo de gabarito não encontrado!")
        return

    df_gabarito = pd.read_csv(caminho_gabarito)
    
    # Treina o extrator com os nomes de colunas que existem no gabarito
    # Isso ajuda a IA a aprender padrões nos nomes (ex: "dt_nasc" geralmente é data)
    extrator.ajustar_nomes_colunas(df_gabarito['nome_coluna'].unique().tolist())

    features_acumuladas = []
    labels_acumulados = []

    print("Iniciando extração de características...")

    # Itera sobre cada arquivo listado no gabarito
    arquivos_unicos = df_gabarito['nome_arquivo'].unique()
    
    for arquivo in arquivos_unicos:
        caminho_completo = os.path.join(caminho_dados, arquivo)
        
        if not os.path.exists(caminho_completo):
            print(f"Aviso: Arquivo {arquivo} não encontrado na pasta de bancos brutos.")
            continue
            
        try:
            # Carrega o CSV usando nosso carregador inteligente
            df_atual, _, _ = carregador.carregar_dados(caminho_completo)
            
            # Filtra o gabarito apenas para as colunas deste arquivo
            gabarito_arquivo = df_gabarito[df_gabarito['nome_arquivo'] == arquivo]
            
            # Para cada coluna documentada no gabarito, extraímos as features
            for _, linha in gabarito_arquivo.iterrows():
                coluna = linha['nome_coluna']
                tipo_real = linha['tipo_semantico']
                
                if coluna in df_atual.columns:
                    # A MÁGICA ACONTECE AQUI:
                    # Transformamos a coluna crua em um dicionário de números (ex: {'numeric_ratio': 0.9, ...})
                    features = extrator.extrair_caracteristicas(df_atual[coluna], coluna)
                    features_acumuladas.append(features)
                    labels_acumulados.append(tipo_real)
        
        except Exception as e:
            print(f"Erro ao processar {arquivo}: {e}")

    # Cria um DataFrame final com todas as features de todas as colunas de todos os arquivos
    df_features = pd.DataFrame(features_acumuladas)
    df_features['target_label'] = labels_acumulados # Adiciona a resposta correta

    # Salva o resultado para ser usado no passo 2
    df_features.to_csv(caminho_saida, index=False)
    print(f"Sucesso! Features extraídas de {len(features_acumuladas)} colunas e salvas em: {caminho_saida}")

if __name__ == "__main__":
    gerar_features_treinamento()