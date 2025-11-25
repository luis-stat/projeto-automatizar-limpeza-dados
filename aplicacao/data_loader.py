import pandas as pd
import numpy as np
import chardet
from typing import Tuple, Optional
import warnings

class DataLoader:
    @staticmethod
    def detect_encoding(file_path: str) -> str:
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
                return encoding
        except Exception:
            return 'utf-8'
            
    @staticmethod
    def detect_separator(file_path: str, encoding: str) -> str:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                first_line = f.readline()
                second_line = f.readline()
                
            lines = [first_line, second_line] if second_line else [first_line]
            
            separator_scores = {}
            for sep in [',', ';', '\t', '|']:
                scores = []
                for line in lines:
                    if line.strip():
                        count = line.count(sep)
                        scores.append(count)
                if scores:
                    separator_scores[sep] = min(scores)
                    
            if separator_scores:
                best_separator = max(separator_scores, key=separator_scores.get)
                if separator_scores[best_separator] > 0:
                    return best_separator
                    
            return ','
        except Exception:
            return ','
            
    def load_data(self, file_path: str) -> Tuple[pd.DataFrame, str, str]:
        try:
            encoding = self.detect_encoding(file_path)
            separator = self.detect_separator(file_path, encoding)
            
            df = pd.read_csv(
                file_path, 
                sep=separator, 
                encoding=encoding, 
                low_memory=False,
                on_bad_lines='skip'
            )
            
            if df.empty:
                raise ValueError("Arquivo CSV está vazio")
                
            return df, separator, encoding
            
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, sep=separator, encoding='latin-1', low_memory=False)
                return df, separator, 'latin-1'
            except Exception as e:
                raise ValueError(f"Erro ao ler arquivo com encoding alternativo: {e}")
                
        except Exception as e:
            raise ValueError(f"Erro ao carregar arquivo: {e}")