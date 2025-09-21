"""
Router Model Training for ARC Expert Selection
Trains a classification model to select appropriate experts for each task
"""

import pandas as pd
import numpy as np
import pickle
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import logging
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARCRouterTrainer:
    """Trains and evaluates router models for expert selection"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.models = {}
        self.best_model = None
        self.feature_names = None
    
    def load_data(self, csv_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare training data"""
        logger.info(f"Loading data from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} training samples")
        
        # Feature columns
        feature_cols = [
            'input_height', 'input_width', 'output_height', 'output_width',
            'size_ratio', 'input_colors', 'output_colors', 'grid_area',
            'complexity_score', 'training_examples'
        ]
        
        # Boolean features
        bool_features = ['color_change', 'shape_change']
        
        # Prepare features
        X = df[feature_cols + bool_features].copy()
        
        # Convert boolean features to int
        for col in bool_features:
            X[col] = X[col].astype(int)
        
        # Target variable
        y = df['pattern_type'].values
        
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Features: {self.feature_names}")
        logger.info(f"Target classes: {np.unique(y)}")
        
        return df, X.values, y
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data for training"""
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Training set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
    
    def train_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train multiple router models"""
        
        models_config = {
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.15],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        }
        
        trained_models = {}
        
        for model_name, config in models_config.items():
            logger.info(f"Training {model_name}...")
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            trained_models[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'grid_search': grid_search
            }
            
            logger.info(f"{model_name} best score: {grid_search.best_score_:.4f}")
            logger.info(f"{model_name} best params: {grid_search.best_params_}")
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Evaluate all trained models"""
        
        evaluation_results = {}
        
        for model_name, model_info in self.models.items():
            model = model_info['model']
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Classification report
            target_names = self.label_encoder.classes_
            class_report = classification_report(
                y_test, y_pred, 
                target_names=target_names,
                output_dict=True
            )
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_names, 
                    model.feature_importances_
                ))
            
            evaluation_results[model_name] = {
                'accuracy': accuracy,
                'classification_report': class_report,
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'prediction_probabilities': y_pred_proba
            }
            
            logger.info(f"{model_name} test accuracy: {accuracy:.4f}")
        
        return evaluation_results
    
    def select_best_model(self, evaluation_results: Dict[str, Dict]) -> str:
        """Select the best performing model"""
        
        best_accuracy = 0
        best_model_name = None
        
        for model_name, results in evaluation_results.items():
            accuracy = results['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]['model']
        logger.info(f"Best model: {best_model_name} (accuracy: {best_accuracy:.4f})")
        
        return best_model_name
    
    def create_visualizations(self, evaluation_results: Dict[str, Dict], y_test: np.ndarray):
        """Create visualization plots"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        model_names = list(evaluation_results.keys())
        accuracies = [results['accuracy'] for results in evaluation_results.values()]
        
        axes[0, 0].bar(model_names, accuracies)
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        # Confusion matrix for best model
        best_model_name = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])[0]
        best_predictions = evaluation_results[best_model_name]['predictions']
        
        cm = confusion_matrix(y_test, best_predictions)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], 
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        axes[0, 1].set_title(f'Confusion Matrix - {best_model_name}')
        
        # Feature importance for best model
        best_importance = evaluation_results[best_model_name]['feature_importance']
        if best_importance:
            features = list(best_importance.keys())
            importances = list(best_importance.values())
            
            # Sort by importance
            sorted_idx = np.argsort(importances)
            features = [features[i] for i in sorted_idx]
            importances = [importances[i] for i in sorted_idx]
            
            axes[1, 0].barh(features, importances)
            axes[1, 0].set_title(f'Feature Importance - {best_model_name}')
        
        # Class distribution
        target_names = self.label_encoder.classes_
        class_counts = [np.sum(y_test == i) for i in range(len(target_names))]
        
        axes[1, 1].pie(class_counts, labels=target_names, autopct='%1.1f%%')
        axes[1, 1].set_title('Test Set Class Distribution')
        
        plt.tight_layout()
        plt.savefig('router_model_evaluation.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("✅ Saved visualization to router_model_evaluation.png")
    
    def save_model(self, model_name: str = "best_router_model"):
        """Save the best model and preprocessing components"""
        
        if self.best_model is None:
            raise ValueError("No best model selected. Run evaluation first.")
        
        # Save model
        joblib.dump(self.best_model, f'{model_name}.pkl')
        
        # Save preprocessing components
        joblib.dump(self.scaler, f'{model_name}_scaler.pkl')
        joblib.dump(self.label_encoder, f'{model_name}_label_encoder.pkl')
        
        # Save feature names
        with open(f'{model_name}_features.json', 'w') as f:
            json.dump(self.feature_names, f)
        
        logger.info(f"✅ Saved model components:")
        logger.info(f"   - {model_name}.pkl")
        logger.info(f"   - {model_name}_scaler.pkl") 
        logger.info(f"   - {model_name}_label_encoder.pkl")
        logger.info(f"   - {model_name}_features.json")

class ARCRouterPredictor:
    """Inference class for the trained router"""
    
    def __init__(self, model_name: str = "best_router_model"):
        self.model = joblib.load(f'{model_name}.pkl')
        self.scaler = joblib.load(f'{model_name}_scaler.pkl')
        self.label_encoder = joblib.load(f'{model_name}_label_encoder.pkl')
        
        with open(f'{model_name}_features.json', 'r') as f:
            self.feature_names = json.load(f)
    
    def predict_expert(self, task_features: Dict[str, Any]) -> Tuple[str, float]:
        """Predict the best expert for a task"""
        
        # Prepare features in the correct order
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in ['color_change', 'shape_change']:
                feature_vector.append(int(task_features[feature_name]))
            else:
                feature_vector.append(task_features[feature_name])
        
        # Scale features
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        # Convert back to pattern type
        pattern_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction]
        
        return pattern_type, confidence
    
    def get_expert_recommendations(self, task_features: Dict[str, Any], top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top-k expert recommendations"""
        
        # Prepare features
        feature_vector = []
        for feature_name in self.feature_names:
            if feature_name in ['color_change', 'shape_change']:
                feature_vector.append(int(task_features[feature_name]))
            else:
                feature_vector.append(task_features[feature_name])
        
        # Scale features
        X = np.array(feature_vector).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities for all classes
        prediction_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get top-k predictions
        top_indices = np.argsort(prediction_proba)[-top_k:][::-1]
        
        recommendations = []
        for idx in top_indices:
            pattern_type = self.label_encoder.inverse_transform([idx])[0]
            confidence = prediction_proba[idx]
            recommendations.append((pattern_type, confidence))
        
        return recommendations

def main():
    """Main training function"""
    
    print("🤖 Training ARC Router Model for Expert Selection")
    print("=" * 55)
    
    # Initialize trainer
    trainer = ARCRouterTrainer()
    
    # Load data
    df, X, y = trainer.load_data('arc_router_training_data.csv')
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test = trainer.prepare_data(X, y)
    
    # Train models
    trained_models = trainer.train_models(X_train_scaled, y_train)
    
    # Evaluate models
    evaluation_results = trainer.evaluate_models(X_test_scaled, y_test)
    
    # Select best model
    best_model_name = trainer.select_best_model(evaluation_results)
    
    # Create visualizations
    trainer.create_visualizations(evaluation_results, y_test)
    
    # Save best model
    trainer.save_model()
    
    # Print detailed results
    print("\\n📊 Model Evaluation Results:")
    print("-" * 40)
    
    for model_name, results in evaluation_results.items():
        print(f"\\n{model_name.upper()}:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        
        if results['feature_importance']:
            print("  Top 5 Important Features:")
            sorted_features = sorted(
                results['feature_importance'].items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            for feature, importance in sorted_features[:5]:
                print(f"    {feature}: {importance:.4f}")
    
    # Test the predictor
    print("\\n🔮 Testing Router Predictor:")
    print("-" * 30)
    
    predictor = ARCRouterPredictor()
    
    # Test with a sample task
    sample_features = {
        'input_height': 3, 'input_width': 3,
        'output_height': 9, 'output_width': 9,
        'size_ratio': 9.0, 'input_colors': 4, 'output_colors': 4,
        'grid_area': 9, 'complexity_score': 0.5, 'training_examples': 3,
        'color_change': False, 'shape_change': True
    }
    
    predicted_expert, confidence = predictor.predict_expert(sample_features)
    recommendations = predictor.get_expert_recommendations(sample_features)
    
    print(f"Sample task features: {sample_features}")
    print(f"Predicted expert: {predicted_expert} (confidence: {confidence:.3f})")
    print("Top 3 recommendations:")
    for i, (expert, conf) in enumerate(recommendations, 1):
        print(f"  {i}. {expert}: {conf:.3f}")
    
    print("\\n🎉 Router Model Training Complete!")
    print(f"✅ Best model: {best_model_name}")
    print(f"✅ Test accuracy: {evaluation_results[best_model_name]['accuracy']:.4f}")

if __name__ == "__main__":
    main()