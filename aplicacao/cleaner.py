import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from rapidfuzz import fuzz, process
from datetime import datetime

class LimpadorDados:
    def __init__(self):
        self.relatorio_limpeza = {}
        
    def para_titulo_br(self, texto: str) -> str:
        """Converte strings para Title Case (iniciais maiúsculas), respeitando preposições pt-BR."""
        if pd.isna(texto) or not isinstance(texto, str):
            return texto
        palavras_pequenas = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com'}
        palavras = texto.split()
        palavras_titulo = []
        for i, palavra in enumerate(palavras):
            if i == 0 or palavra.lower() not in palavras_pequenas:
                if len(palavra) > 1:
                    palavras_titulo.append(palavra[0].upper() + palavra[1:].lower())
                else:
                    palavras_titulo.append(palavra.upper())
            else:
                palavras_titulo.append(palavra.lower())
        return ' '.join(palavras_titulo)
        
    def padronizar_data(self, data_str: Any) -> Any:
        """Tenta converter diversos formatos de data para o padrão dd/mm/aaaa."""
        if pd.isna(data_str):
            return data_str
        str_data = str(data_str).strip()
        if not str_data:
            return np.nan
        try:
            obj_data = pd.to_datetime(str_data, dayfirst=True, errors='coerce')
            if pd.isna(obj_data):
                return np.nan
            return obj_data.strftime('%d/%m/%Y')
        except Exception:
            return np.nan
            
    def detectar_valores_frequentes(self, serie: pd.Series, limiar: float = 0.05) -> Tuple[List[Any], List[Any]]:
        """Separa valores muito comuns (corretos) de valores raros (possíveis erros)."""
        contagem_valores = serie.value_counts()
        total_nao_nulos = len(serie.dropna())
        if total_nao_nulos == 0:
            return [], []
        valores_frequentes = []
        valores_raros = []
        for valor, contagem in contagem_valores.items():
            frequencia = contagem / total_nao_nulos
            if frequencia >= limiar:
                valores_frequentes.append(valor)
            else:
                valores_raros.append(valor)
        return valores_frequentes, valores_raros
        
    def correcao_fuzzy(self, serie: pd.Series, limiar_similaridade: float = 85) -> Tuple[pd.Series, Dict[str, Any]]:
        """Corrige erros de digitação comparando valores raros com valores frequentes."""
        if serie.dtype != 'object':
            return serie, {}
        serie_limpa = serie.copy()
        valores_frequentes, valores_raros = self.detectar_valores_frequentes(serie_limpa)
        if not valores_frequentes or not valores_raros:
            return serie_limpa, {}
            
        correcoes = {}
        linhas_corrigidas = 0
        linhas_removidas = 0
        
        for valor_raro in valores_raros:
            if pd.isna(valor_raro):
                continue
            melhor_match, pontuacao, _ = process.extractOne(
                str(valor_raro), 
                [str(vf) for vf in valores_frequentes], 
                scorer=fuzz.token_sort_ratio
            )
            if pontuacao >= limiar_similaridade:
                valor_freq_original = valores_frequentes[[str(vf) for vf in valores_frequentes].index(melhor_match)]
                mascara = serie_limpa == valor_raro
                serie_limpa[mascara] = valor_freq_original
                correcoes[str(valor_raro)] = str(valor_freq_original)
                linhas_corrigidas += mascara.sum()
            else:
                # Se não for similar o suficiente, considera ruído e remove
                mascara = serie_limpa == valor_raro
                serie_limpa[mascara] = np.nan
                linhas_removidas += mascara.sum()
                
        estatisticas = {
            'linhas_corrigidas': linhas_corrigidas,
            'linhas_removidas': linhas_removidas,
            'correcoes_feitas': correcoes
        }
        return serie_limpa, estatisticas
        
    def limpar_coluna(self, serie: pd.Series, tipo_coluna: str, nome_coluna: str, limiar_similaridade: float = 85) -> Tuple[pd.Series, Dict[str, Any]]:
        """Aplica regras de limpeza específicas baseadas no tipo semântico da coluna."""
        stats_limpeza = {
            'original_nao_nulos': serie.notna().sum(),
            'linhas_corrigidas': 0,
            'linhas_removidas': 0,
            'datas_formatadas': 0
        }
        serie_limpa = serie.copy()
        
        if tipo_coluna in ['TEXTO_LIVRE', 'CATEGORICO_NOMINAL']:
            mascara_texto = serie_limpa.notna() & (serie_limpa.astype(str).str.strip() != '')
            serie_limpa[mascara_texto] = serie_limpa[mascara_texto].apply(self.para_titulo_br)
            serie_limpa, stats_fuzzy = self.correcao_fuzzy(serie_limpa, limiar_similaridade)
            stats_limpeza.update(stats_fuzzy)
            
        elif tipo_coluna == 'DATA_HORA':
            mascara_data = serie_limpa.notna()
            serie_limpa[mascara_data] = serie_limpa[mascara_data].apply(self.padronizar_data)
            stats_limpeza['datas_formatadas'] = mascara_data.sum()
            
            nova_contagem_nulos = serie_limpa.isna().sum()
            contagem_nulos_original = serie.isna().sum()
            stats_limpeza['linhas_removidas'] += (nova_contagem_nulos - contagem_nulos_original)
            
        stats_limpeza['final_nao_nulos'] = serie_limpa.notna().sum()
        return serie_limpa, stats_limpeza
        
    def limpar_dataset(self, df: pd.DataFrame, previsoes_tipo: Dict[str, str], limiar_similaridade: float = 85) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Itera sobre todas as colunas do DataFrame e aplica a limpeza."""
        df_limpo = df.copy()
        relatorio_geral = {
            'colunas_limpas': {},
            'total_linhas_corrigidas': 0,
            'total_linhas_removidas': 0,
            'total_datas_formatadas': 0
        }
        for coluna in df.columns:
            if coluna in previsoes_tipo:
                tipo_coluna = previsoes_tipo[coluna]
                serie_limpa, stats_col = self.limpar_coluna(
                    df[coluna], tipo_coluna, coluna, limiar_similaridade
                )
                df_limpo[coluna] = serie_limpa
                relatorio_geral['colunas_limpas'][coluna] = stats_col
                relatorio_geral['total_linhas_corrigidas'] += stats_col.get('linhas_corrigidas', 0)
                relatorio_geral['total_linhas_removidas'] += stats_col.get('linhas_removidas', 0)
                relatorio_geral['total_datas_formatadas'] += stats_col.get('datas_formatadas', 0)
                
        self.relatorio_limpeza = relatorio_geral
        return df_limpo, relatorio_geral