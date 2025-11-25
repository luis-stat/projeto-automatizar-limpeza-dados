import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
from typing import Dict, Any
import warnings

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader
from feature_extractor import MetaFeatureExtractor
from cleaner import DataCleaner

class SemanticTypeInferenceApp:
    def __init__(self):
        self.data_loader = DataLoader()
        self.feature_extractor = MetaFeatureExtractor()
        self.cleaner = DataCleaner()
        self.model = None
        self.label_encoder = None
        
        self.load_model()
        
    def load_model(self):
        try:
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'modelos_salvos',
                'semantic_type_classifier.pkl'
            )
            
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
                
            self.model = model_data['model']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data.get('feature_columns', [])
            
            print(f"Modelo carregado: {len(self.feature_columns)} features, {len(self.label_encoder.classes_)} classes")
            
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            st.stop()
            
    def predict_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        predictions = {}
        feature_list = []
        column_names = []
        
        for column in df.columns:
            features = self.feature_extractor.extract_features(df[column], column)
            feature_list.append(features)
            column_names.append(column)
            
        features_df = pd.DataFrame(feature_list)
        features_df = features_df.fillna(0)
        
        for col in features_df.select_dtypes(include=['object']).columns:
            features_df[col] = pd.to_numeric(features_df[col], errors='coerce')
        features_df = features_df.fillna(0)
        
        try:
            predictions_encoded = self.model.predict(features_df)
            predicted_types = self.label_encoder.inverse_transform(predictions_encoded)
            
            for col_name, pred_type in zip(column_names, predicted_types):
                predictions[col_name] = pred_type
                
        except Exception as e:
            st.error(f"Erro durante a predição: {e}")
            for col_name in column_names:
                predictions[col_name] = 'DESCONHECIDO'
                
        return predictions
        
    def run(self):
        st.set_page_config(
            page_title="Inferência Semântica de Tipos de Dados",
            layout="wide"
        )
        
        st.title("Sistema de Inferência Semântica de Tipos de Dados")
        st.markdown("---")
        
        uploaded_file = st.file_uploader(
            "Carregue seu arquivo CSV",
            type=['csv'],
            help="Suporta arquivos CSV com separador , ou ;"
        )
        
        if uploaded_file is not None:
            try:
                with st.spinner("Processando arquivo..."):
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    df, separator, encoding = self.data_loader.load_data(temp_path)
                    os.remove(temp_path)
                    
                st.success(f"Arquivo carregado com sucesso: {len(df)} linhas × {len(df.columns)} colunas")
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("Pré-visualização dos Dados")
                    st.dataframe(df.head(10), use_container_width=True)
                    
                with col2:
                    st.subheader("Metadados")
                    st.metric("Total de Linhas", len(df))
                    st.metric("Total de Colunas", len(df.columns))
                    st.metric("Separador", separator)
                    st.metric("Encoding", encoding)
                    
                if st.button("Inferir Tipos de Dados e Higienizar", type="primary"):
                    with st.spinner("Analisando tipos de dados..."):
                        type_predictions = self.predict_column_types(df)
                        
                    st.subheader("Resultados da Inferência de Tipos")
                    
                    type_df = pd.DataFrame([
                        {'Coluna': col, 'Tipo Inferido': tipo}
                        for col, tipo in type_predictions.items()
                    ])
                    
                    st.dataframe(type_df, use_container_width=True)
                    
                    with st.spinner("Aplicando limpeza e correção..."):
                        cleaned_df, cleaning_report = self.cleaner.clean_dataset(df, type_predictions)
                        
                    st.subheader("Relatório de Limpeza")
                    
                    total_corrections = cleaning_report['total_rows_corrected']
                    total_removals = cleaning_report['total_rows_removed']
                    total_dates = cleaning_report['total_dates_formatted']
                    
                    if total_corrections > 0:
                        st.info(f"{total_corrections} linhas corrigidas por similaridade fuzzy")
                    if total_removals > 0:
                        st.warning(f"{total_removals} linhas removidas por inconsistência")
                    if total_dates > 0:
                        st.info(f"{total_dates} datas formatadas para padrão brasileiro")
                        
                    if total_corrections == 0 and total_removals == 0 and total_dates == 0:
                        st.success("✅ Nenhuma alteração necessária - dados já estão limpos")
                        
                    st.subheader("Pré-visualização dos Dados Limpos")
                    st.dataframe(cleaned_df.head(10), use_container_width=True)
                    
                    csv_cleaned = cleaned_df.to_csv(index=False, sep=';')
                    
                    st.download_button(
                        label="📥 Download Dataset Limpo",
                        data=csv_cleaned,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv",
                        type="primary"
                    )
                    
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {e}")

if __name__ == "__main__":
    app = SemanticTypeInferenceApp()
    app.run()