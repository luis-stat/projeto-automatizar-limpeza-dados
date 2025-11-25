import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from typing import Dict, Any, List
import warnings

class MetaFeatureExtractor:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50,
            stop_words=None,
            lowercase=True,
            analyzer='char_wb',
            ngram_range=(2, 4)
        )
        self.column_names_fitted = False
        
    def _can_convert_to_numeric(self, value):
        try:
            if pd.isna(value):
                return False
            float(str(value).replace(',', '.'))
            return True
        except (ValueError, TypeError):
            return False
            
    def _can_convert_to_datetime(self, value):
        try:
            if pd.isna(value):
                return False
            pd.to_datetime(str(value), errors='raise', dayfirst=True)
            return True
        except (ValueError, TypeError, pd.errors.ParserError):
            return False
            
    def _get_string_length_stats(self, series):
        str_series = series.astype(str)
        lengths = str_series.str.len()
        return lengths.mean(), lengths.std()
        
    def _extract_pandas_dtype_features(self, dtype):
        dtype_mapping = {
            'object': [1, 0, 0, 0],
            'int64': [0, 1, 0, 0],
            'float64': [0, 0, 1, 0],
            'datetime64[ns]': [0, 0, 0, 1]
        }
        return dtype_mapping.get(str(dtype), [0, 0, 0, 0])
        
    def fit_column_names(self, column_names: List[str]):
        if column_names:
            try:
                self.tfidf_vectorizer.fit(column_names)
                self.column_names_fitted = True
            except Exception as e:
                warnings.warn(f"TF-IDF fitting failed: {e}")
                self.column_names_fitted = False
                
    def extract_features(self, series: pd.Series, column_name: str = "") -> Dict[str, Any]:
        features = {}
        
        series_clean = series.dropna()
        total_count = len(series)
        non_null_count = len(series_clean)
        
        if total_count == 0:
            return self._get_empty_features()
            
        features['non_null_ratio'] = non_null_count / total_count if total_count > 0 else 0
        features['cardinality_ratio'] = series_clean.nunique() / non_null_count if non_null_count > 0 else 0
        
        numeric_count = series_clean.apply(self._can_convert_to_numeric).sum()
        features['numeric_conversion_ratio'] = numeric_count / non_null_count if non_null_count > 0 else 0
        
        datetime_count = series_clean.apply(self._can_convert_to_datetime).sum()
        features['datetime_conversion_ratio'] = datetime_count / non_null_count if non_null_count > 0 else 0
        
        avg_len, std_len = self._get_string_length_stats(series_clean)
        features['avg_string_length'] = avg_len
        features['std_string_length'] = std_len if not np.isnan(std_len) else 0
        
        dtype_features = self._extract_pandas_dtype_features(series.dtype)
        features['dtype_object'], features['dtype_int'], features['dtype_float'], features['dtype_datetime'] = dtype_features
        
        if self.column_names_fitted and column_name:
            try:
                name_embedding = self.tfidf_vectorizer.transform([column_name]).toarray()[0]
                for i, val in enumerate(name_embedding):
                    features[f'name_embedding_{i}'] = val
            except Exception:
                for i in range(50):
                    features[f'name_embedding_{i}'] = 0
        else:
            for i in range(50):
                features[f'name_embedding_{i}'] = 0
                
        return features
        
    def _get_empty_features(self):
        base_features = {
            'non_null_ratio': 0,
            'cardinality_ratio': 0,
            'numeric_conversion_ratio': 0,
            'datetime_conversion_ratio': 0,
            'avg_string_length': 0,
            'std_string_length': 0,
            'dtype_object': 1,
            'dtype_int': 0,
            'dtype_float': 0,
            'dtype_datetime': 0
        }
        for i in range(50):
            base_features[f'name_embedding_{i}'] = 0
        return base_features