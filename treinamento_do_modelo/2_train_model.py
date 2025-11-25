import pandas as pd
import numpy as np
import os
import pickle
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import warnings

class ModelTrainer:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = None
        self.X = None
        
    def load_features(self) -> pd.DataFrame:
        features_path = os.path.join(self.base_dir, 'dados_para_treinamento', 'training_features.csv')
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Arquivo de features não encontrado: {features_path}")
        return pd.read_csv(features_path)
        
    def handle_class_imbalance(self, X: pd.DataFrame, y: pd.Series):
        class_counts = y.value_counts()
        single_instance_classes = class_counts[class_counts == 1].index.tolist()
        
        if single_instance_classes:
            warnings.warn(f"Classes com apenas 1 instância: {single_instance_classes}. Duplicando instâncias.")
            
            additional_X = []
            additional_y = []
            
            for class_name in single_instance_classes:
                class_mask = y == class_name
                class_X = X[class_mask].copy()
                additional_X.append(class_X)
                additional_y.extend([class_name] * len(class_X))
                
            if additional_X:
                X_extra = pd.concat(additional_X, ignore_index=True)
                X = pd.concat([X, X_extra], ignore_index=True)
                y = pd.concat([y, pd.Series(additional_y)], ignore_index=True)
                
        return X, y
        
    def train_model(self):
        features_df = self.load_features()
        
        if 'target' not in features_df.columns:
            raise ValueError("Coluna 'target' não encontrada no dataset de features")
            
        X = features_df.drop(['target', 'nome_arquivo', 'nome_coluna'], axis=1, errors='ignore')
        y = features_df['target']
        
        # Salvar as colunas de features para uso futuro
        self.feature_columns = list(X.columns)
        self.X = X.copy()
        
        X = X.fillna(0)
        
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        X = X.fillna(0)
        
        X, y = self.handle_class_imbalance(X, y)
        
        y_encoded = self.label_encoder.fit_transform(y)
        
        unique_classes = len(np.unique(y_encoded))
        
        if unique_classes < 2:
            raise ValueError(f"Número insuficiente de classes para treino: {unique_classes}")
            
        stratification = y if unique_classes > 1 else None
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42,
                stratify=stratification
            )
        except ValueError as e:
            warnings.warn(f"Erro na estratificação: {e}. Prosseguindo sem estratificação.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, 
                test_size=0.2, 
                random_state=42
            )
            
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            max_depth=-1,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Treino compatível com diferentes versões do LightGBM
        try:
            # Versões mais recentes
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='logloss',
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                verbose=False
            )
        except TypeError:
            try:
                # Tentativa com early_stopping_rounds (versões intermediárias)
                self.model.fit(
                    X_train, y_train,
                    eval_set=[(X_test, y_test)],
                    eval_metric='logloss',
                    early_stopping_rounds=50,
                    verbose=False
                )
            except TypeError:
                # Fallback: treino simples sem early stopping
                warnings.warn("Early stopping não suportado. Treinando sem validação.")
                self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Acurácia do modelo: {accuracy:.4f}")
        print("\nRelatório de Classificação:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Features Mais Importantes:")
            print(feature_importance.head(10))
        
        return X_test, y_test, y_pred
        
    def save_model(self):
        if self.model is None:
            raise ValueError("Modelo não foi treinado. Execute train_model primeiro.")
            
        modelos_dir = os.path.join(self.base_dir, 'modelos_salvos')
        os.makedirs(modelos_dir, exist_ok=True)
        
        model_path = os.path.join(modelos_dir, 'semantic_type_classifier.pkl')
        
        # Usar self.feature_columns que foi definido no train_model
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns if self.feature_columns is not None else []
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"Modelo salvo em: {model_path}")
        print(f"Número de features salvas: {len(self.feature_columns) if self.feature_columns else 0}")

if __name__ == "__main__":
    try:
        trainer = ModelTrainer()
        X_test, y_test, y_pred = trainer.train_model()
        trainer.save_model()
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)