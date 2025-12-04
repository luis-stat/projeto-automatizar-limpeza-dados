import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from rapidfuzz import fuzz, process
from datetime import datetime

class DataCleaner:
    def __init__(self):
        self.cleaning_report = {}
        
    def to_title_case_br(self, text: str) -> str:
        if pd.isna(text) or not isinstance(text, str):
            return text
        small_words = {'de', 'da', 'do', 'das', 'dos', 'e', 'em', 'para', 'com'}
        words = text.split()
        title_words = []
        for i, word in enumerate(words):
            if i == 0 or word.lower() not in small_words:
                if len(word) > 1:
                    title_words.append(word[0].upper() + word[1:].lower())
                else:
                    title_words.append(word.upper())
            else:
                title_words.append(word.lower())
        return ' '.join(title_words)
        
    def standardize_date(self, date_str: Any) -> Any:
        if pd.isna(date_str):
            return date_str
        str_date = str(date_str).strip()
        if not str_date:
            return np.nan
        try:
            date_obj = pd.to_datetime(str_date, dayfirst=True, errors='coerce')
            if pd.isna(date_obj):
                return np.nan
            return date_obj.strftime('%d/%m/%Y')
        except Exception:
            return np.nan
            
    def detect_frequent_values(self, series: pd.Series, threshold: float = 0.05) -> Tuple[List[Any], List[Any]]:
        value_counts = series.value_counts()
        total_non_null = len(series.dropna())
        if total_non_null == 0:
            return [], []
        frequent_values = []
        rare_values = []
        for value, count in value_counts.items():
            frequency = count / total_non_null
            if frequency >= threshold:
                frequent_values.append(value)
            else:
                rare_values.append(value)
        return frequent_values, rare_values
        
    def fuzzy_correction(self, series: pd.Series, similarity_threshold: float = 85) -> Tuple[pd.Series, Dict[str, Any]]:
        if series.dtype != 'object':
            return series, {}
        series_clean = series.copy()
        frequent_values, rare_values = self.detect_frequent_values(series_clean)
        if not frequent_values or not rare_values:
            return series_clean, {}
        corrections = {}
        rows_corrected = 0
        rows_removed = 0
        for rare_value in rare_values:
            if pd.isna(rare_value):
                continue
            best_match, score, _ = process.extractOne(
                str(rare_value), 
                [str(fv) for fv in frequent_values], 
                scorer=fuzz.token_sort_ratio
            )
            if score >= similarity_threshold:
                original_freq_value = frequent_values[[str(fv) for fv in frequent_values].index(best_match)]
                mask = series_clean == rare_value
                series_clean[mask] = original_freq_value
                corrections[str(rare_value)] = str(original_freq_value)
                rows_corrected += mask.sum()
            else:
                mask = series_clean == rare_value
                series_clean[mask] = np.nan
                rows_removed += mask.sum()
        stats = {
            'rows_corrected': rows_corrected,
            'rows_removed': rows_removed,
            'corrections_made': corrections
        }
        return series_clean, stats
        
    def clean_column(self, series: pd.Series, column_type: str, column_name: str, similarity_threshold: float = 85) -> Tuple[pd.Series, Dict[str, Any]]:
        cleaning_stats = {
            'original_non_null': series.notna().sum(),
            'rows_corrected': 0,
            'rows_removed': 0,
            'dates_formatted': 0
        }
        cleaned_series = series.copy()
        if column_type in ['TEXTO_LIVRE', 'CATEGORICO_NOMINAL']:
            text_mask = cleaned_series.notna() & (cleaned_series.astype(str).str.strip() != '')
            cleaned_series[text_mask] = cleaned_series[text_mask].apply(self.to_title_case_br)
            cleaned_series, fuzzy_stats = self.fuzzy_correction(cleaned_series, similarity_threshold)
            cleaning_stats.update(fuzzy_stats)
        elif column_type == 'DATA_HORA':
            date_mask = cleaned_series.notna()
            cleaned_series[date_mask] = cleaned_series[date_mask].apply(self.standardize_date)
            cleaning_stats['dates_formatted'] = date_mask.sum()
            new_null_count = cleaned_series.isna().sum()
            original_null_count = series.isna().sum()
            cleaning_stats['rows_removed'] += (new_null_count - original_null_count)
        cleaning_stats['final_non_null'] = cleaned_series.notna().sum()
        return cleaned_series, cleaning_stats
        
    def clean_dataset(self, df: pd.DataFrame, type_predictions: Dict[str, str], similarity_threshold: float = 85) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        cleaned_df = df.copy()
        overall_report = {
            'columns_cleaned': {},
            'total_rows_corrected': 0,
            'total_rows_removed': 0,
            'total_dates_formatted': 0
        }
        for column in df.columns:
            if column in type_predictions:
                column_type = type_predictions[column]
                cleaned_series, col_stats = self.clean_column(
                    df[column], column_type, column, similarity_threshold
                )
                cleaned_df[column] = cleaned_series
                overall_report['columns_cleaned'][column] = col_stats
                overall_report['total_rows_corrected'] += col_stats.get('rows_corrected', 0)
                overall_report['total_rows_removed'] += col_stats.get('rows_removed', 0)
                overall_report['total_dates_formatted'] += col_stats.get('dates_formatted', 0)
        self.cleaning_report = overall_report
        return cleaned_df, overall_report