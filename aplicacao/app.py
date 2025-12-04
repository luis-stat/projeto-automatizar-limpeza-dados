import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle
import csv
from datetime import datetime

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
        self.feature_columns = []
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
        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            st.stop()

    def save_feedback(self, column_name, predicted_type, correct_type, file_name):
        feedback_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'dados_para_treinamento',
            'feedback_loop.csv'
        )
        file_exists = os.path.isfile(feedback_file)
        with open(feedback_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['data_feedback', 'nome_arquivo', 'nome_coluna', 'tipo_predito', 'tipo_correto'])
            writer.writerow([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), file_name, column_name, predicted_type, correct_type])
            
    def predict_column_types(self, df: pd.DataFrame) -> dict:
        predictions = {}
        feature_list = []
        column_names = []
        for column in df.columns:
            features = self.feature_extractor.extract_features(df[column], column)
            feature_list.append(features)
            column_names.append(column)
        features_df = pd.DataFrame(feature_list)
        features_df = features_df.fillna(0)
        
        missing_cols = set(self.feature_columns) - set(features_df.columns)
        for c in missing_cols:
            features_df[c] = 0
        features_df = features_df[self.feature_columns]
        
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
        st.set_page_config(page_title="Inferência Semântica", layout="wide")
        st.title("Sistema de Inferência e Limpeza de Dados")
        
        with st.sidebar:
            st.header("Configurações de Limpeza")
            fuzzy_threshold = st.slider(
                "Sensibilidade da Correção (Fuzzy)", 
                min_value=50, max_value=100, value=85,
                help="Valores maiores exigem mais similaridade para corrigir erros de digitação."
            )
            st.markdown("---")
            st.header("Feedback Loop")
            st.info("Suas correções manuais serão salvas para re-treinar o modelo futuramente.")

        uploaded_file = st.file_uploader("Carregue seu arquivo CSV", type=['csv'])
        
        if uploaded_file is not None:
            if 'df_raw' not in st.session_state or st.session_state.get('uploaded_filename') != uploaded_file.name:
                try:
                    with st.spinner("Carregando arquivo..."):
                        temp_path = f"temp_{uploaded_file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        df, separator, encoding = self.data_loader.load_data(temp_path)
                        os.remove(temp_path)
                        st.session_state['df_raw'] = df
                        st.session_state['uploaded_filename'] = uploaded_file.name
                        st.session_state['type_predictions'] = self.predict_column_types(df)
                except Exception as e:
                    st.error(f"Erro: {e}")
                    return
            
            df = st.session_state['df_raw']
            type_predictions = st.session_state['type_predictions']
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.subheader("Pré-visualização")
                st.dataframe(df.head(), use_container_width=True)
            with col2:
                st.subheader("Tipos Identificados")
                corrections = {}
                for col in df.columns:
                    pred = type_predictions.get(col, 'DESCONHECIDO')
                    possible_types = list(self.label_encoder.classes_) if self.label_encoder else [pred]
                    new_type = st.selectbox(
                        f"{col}", 
                        options=possible_types, 
                        index=possible_types.index(pred) if pred in possible_types else 0,
                        key=f"sel_{col}"
                    )
                    if new_type != pred:
                        corrections[col] = (pred, new_type)
                        type_predictions[col] = new_type

                if corrections:
                    if st.button("Salvar Feedback de Tipos"):
                        for col, (old, new) in corrections.items():
                            self.save_feedback(col, old, new, uploaded_file.name)
                        st.success("Feedback salvo com sucesso!")

            if st.button("Executar Limpeza", type="primary"):
                with st.spinner("Limpando dados..."):
                    cleaned_df, report = self.cleaner.clean_dataset(df, type_predictions, fuzzy_threshold)
                    
                st.subheader("Resultado da Limpeza")
                st.dataframe(cleaned_df, use_container_width=True)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Linhas Corrigidas", report['total_rows_corrected'])
                c2.metric("Linhas Removidas", report['total_rows_removed'])
                c3.metric("Datas Formatadas", report['total_dates_formatted'])
                
                st.download_button(
                    label="Baixar CSV Limpo",
                    data=cleaned_df.to_csv(index=False, sep=';'),
                    file_name="cleaned_dataset.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    app = SemanticTypeInferenceApp()
    app.run()