import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Dict, Any, List
import warnings

class MetaFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(2, 4)
        )
        self.column_names_fitted = False
        
    def _can_convert_to_numeric(self, value):
        try:
            if pd.isna(value): return False
            float(str(value).replace(',', '.'))
            return True
        except: return False
            
    def _can_convert_to_datetime(self, value):
        try:
            if pd.isna(value): return False
            pd.to_datetime(str(value), errors='raise', dayfirst=True)
            return True
        except: return False
        
    def fit_column_names(self, column_names: List[str]):
        if column_names:
            try:
                self.tfidf_vectorizer.fit(column_names)
                self.column_names_fitted = True
            except: self.column_names_fitted = False
                
    def extract_features(self, series: pd.Series, column_name: str = "") -> Dict[str, Any]:
        features = {}
        series_clean = series.dropna()
        total = len(series)
        non_null = len(series_clean)
        
        features['non_null_ratio'] = non_null / total if total > 0 else 0
        features['cardinality_ratio'] = series_clean.nunique() / non_null if non_null > 0 else 0
        
        sample = series_clean.sample(min(len(series_clean), 1000), random_state=42) if non_null > 0 else series_clean
        
        features['numeric_ratio'] = sample.apply(self._can_convert_to_numeric).sum() / len(sample) if len(sample) > 0 else 0
        features['date_ratio'] = sample.apply(self._can_convert_to_datetime).sum() / len(sample) if len(sample) > 0 else 0
        
        str_lens = sample.astype(str).str.len()
        features['avg_len'] = str_lens.mean() if not str_lens.empty else 0
        
        dtype_map = {'object': [1,0,0], 'int64': [0,1,0], 'float64': [0,0,1]}
        d_feats = dtype_map.get(str(series.dtype), [0,0,0])
        features['is_obj'], features['is_int'], features['is_float'] = d_feats
        
        if self.column_names_fitted and column_name:
            try:
                emb = self.tfidf_vectorizer.transform([column_name]).toarray()[0]
                for i, v in enumerate(emb): features[f'nm_emb_{i}'] = v
            except:
                for i in range(50): features[f'nm_emb_{i}'] = 0
        else:
            for i in range(50): features[f'nm_emb_{i}'] = 0
                
        return features