import os
import joblib
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CNNExecutionPredictor(nn.Module):
    """Neural Network model for CNN execution time prediction"""

    def __init__(self, input_dim=7, hidden_dim=128, dropout_rate=0.3):
        super(CNNExecutionPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.ReLU()  # Ensure positive predictions for execution time
        )

    def forward(self, x):
        return self.network(x)


class ModelLoader:
    """Unified model loader for different ML models"""

    def __init__(self, models_dir: str = "../CNN_Algorithms"):
        # Handle both relative and absolute paths
        if os.path.isabs(models_dir):
            self.models_dir = Path(models_dir)
        else:
            # For relative paths, resolve from the current working directory
            self.models_dir = Path(os.getcwd()) / models_dir
            # If that doesn't exist, try from the app directory
            if not self.models_dir.exists():
                app_dir = Path(__file__).parent.parent.parent
                self.models_dir = app_dir / models_dir.lstrip("../")

        logger.info(f"Using models directory: {self.models_dir}")
        logger.info(f"Models directory exists: {self.models_dir.exists()}")
        self.loaded_models = {}
        self.model_metadata = {}

    def load_pytorch_model(self, model_path: str, model_name: str) -> bool:
        """Load PyTorch neural network model"""
        try:
            # Initialize model architecture
            model = CNNExecutionPredictor()

            # Load the state dict
            state_dict = torch.load(model_path, map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            self.loaded_models[model_name] = {
                'model': model,
                'type': 'pytorch',
                'input_features': ['batch_size', 'input_channels', 'input_height', 'input_width',
                                   'output_channels', 'kernel_size', 'stride']
            }

            logger.info(f"Successfully loaded PyTorch model: {model_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to load PyTorch model {model_name}: {str(e)}")
            return False

    def load_sklearn_model(self, model_path: str, model_name: str, model_type: str) -> bool:
        """Load scikit-learn based models (XGBoost, Random Forest, Polynomial Regression)"""
        try:
            model = joblib.load(model_path)

            # Determine input features based on model type
            if 'xgboost' in model_name.lower() or 'xgb' in model_name.lower():
                input_features = ['batch_size', 'input_channels', 'input_height', 'input_width',
                                  'output_channels', 'kernel_size', 'stride']
            elif 'random_forest' in model_name.lower() or 'rf' in model_name.lower():
                input_features = ['batch_size', 'input_channels', 'input_height', 'input_width',
                                  'output_channels', 'kernel_size', 'stride']
            elif 'polynomial' in model_name.lower() or 'lasso' in model_name.lower() or 'ridge' in model_name.lower():
                # Polynomial models might have more features due to polynomial expansion
                input_features = ['batch_size', 'input_channels', 'input_height', 'input_width',
                                  'output_channels', 'kernel_size', 'stride', 'padding']
            else:
                input_features = ['batch_size', 'input_channels', 'input_height', 'input_width',
                                  'output_channels', 'kernel_size', 'stride']

            self.loaded_models[model_name] = {
                'model': model,
                'type': model_type,
                'input_features': input_features
            }

            logger.info(f"Successfully loaded sklearn model: {model_name}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to load sklearn model {model_name}: {str(e)}")
            return False

    def auto_discover_models(self) -> Dict[str, List[str]]:
        """Automatically discover available trained models in the project"""
        discovered_models = {
            'pytorch': [],
            'xgboost': [],
            'random_forest': [],
            'polynomial': []
        }

        # Search for PyTorch models
        for pth_file in self.models_dir.rglob("*.pth"):
            if "best_model" in str(pth_file):
                discovered_models['pytorch'].append(str(pth_file))

        # Search for joblib models
        for joblib_file in self.models_dir.rglob("*.joblib"):
            file_str = str(joblib_file).lower()

            # Skip scalers and preprocessing objects
            if any(skip_word in file_str for skip_word in ['scaler', 'preprocessor', 'encoder', 'transformer']):
                continue

            if "xgb" in file_str or "xgboost" in file_str:
                discovered_models['xgboost'].append(str(joblib_file))
            elif "rf" in file_str or "random_forest" in file_str:
                discovered_models['random_forest'].append(str(joblib_file))
            elif any(keyword in file_str for keyword in ['lasso', 'ridge', 'polynomial', 'elasticnet']):
                # Only include best models, not individual folds or scalers
                if 'best' in file_str and 'model' in file_str:
                    discovered_models['polynomial'].append(str(joblib_file))

        return discovered_models

    def load_all_available_models(self) -> Dict[str, bool]:
        """Load all discovered models"""
        discovered = self.auto_discover_models()
        load_results = {}

        # Load PyTorch models
        for model_path in discovered['pytorch']:
            model_name = f"nn_{Path(model_path).parent.parent.name}"
            load_results[model_name] = self.load_pytorch_model(
                model_path, model_name)

        # Load XGBoost models
        for model_path in discovered['xgboost']:
            model_name = f"xgb_{Path(model_path).parent.parent.name}"
            load_results[model_name] = self.load_sklearn_model(
                model_path, model_name, 'xgboost')

        # Load Random Forest models
        for model_path in discovered['random_forest']:
            model_name = f"rf_{Path(model_path).parent.parent.name}"
            load_results[model_name] = self.load_sklearn_model(
                model_path, model_name, 'random_forest')

        # Load Polynomial models
        for model_path in discovered['polynomial']:
            model_name = f"poly_{Path(model_path).stem}"
            load_results[model_name] = self.load_sklearn_model(
                model_path, model_name, 'polynomial')

        return load_results

    def predict(self, model_name: str, input_features: Dict[str, float]) -> Dict[str, Any]:
        """Make prediction using specified model"""
        try:
            if model_name not in self.loaded_models:
                raise ValueError(
                    f"Model {model_name} not loaded. Available models: {list(self.loaded_models.keys())}")

            model_info = self.loaded_models[model_name]
            model = model_info['model']

            # Prepare input array based on model type
            if model_info['type'] == 'polynomial':
                # Polynomial models expect these 9 features in this order:
                # ['Batch_Size', 'Input_Size', 'In_Channels', 'Out_Channels', 'Kernel_Size', 'Stride', 'Padding', 'Algorithm_direct_cpu', 'Algorithm_gemm']

                # Map from API features to training features
                batch_size = input_features.get('batch_size', 32)
                input_size = input_features.get(
                    'input_height', 224) * input_features.get('input_width', 224)
                in_channels = input_features.get('input_channels', 3)
                out_channels = input_features.get('output_channels', 64)
                kernel_size = input_features.get('kernel_size', 3)
                stride = input_features.get('stride', 1)
                padding = input_features.get(
                    'padding', 0)  # Default padding is 0

                # For the algorithm columns, we'll assume 'gemm' algorithm (1, 0)
                # This can be made configurable later
                algorithm_direct_cpu = 0
                algorithm_gemm = 1

                feature_values = [
                    batch_size, input_size, in_channels, out_channels,
                    kernel_size, stride, padding, algorithm_direct_cpu, algorithm_gemm
                ]

            else:
                # For other models (pytorch, xgboost, random_forest), use the standard 7 features
                feature_values = []
                for feature in model_info['input_features']:
                    if feature not in input_features:
                        raise ValueError(
                            f"Missing required feature: {feature}")
                    feature_values.append(input_features[feature])

            input_array = np.array([feature_values])
            logger.info(
                f"Making prediction with {model_name}, input shape: {input_array.shape}, values: {feature_values}")

            # Make prediction based on model type
            if model_info['type'] == 'pytorch':
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(input_array)
                    prediction = model(input_tensor).numpy().flatten()[0]
            elif model_info['type'] in ['polynomial', 'xgboost', 'random_forest']:
                # For sklearn-based models
                prediction = model.predict(input_array)[0]
            else:
                raise ValueError(f"Unknown model type: {model_info['type']}")

            # Ensure prediction is a positive number (execution time can't be negative)
            prediction = max(0, float(prediction))

            logger.info(
                f"Prediction successful for {model_name}: {prediction}")

            return {
                'prediction': prediction,
                'model_name': model_name,
                'model_type': model_info['type'],
                'status': 'success'
            }

        except Exception as e:
            logger.error(
                f"Prediction failed for {model_name}: {str(e)}", exc_info=True)
            return {
                'prediction': None,
                'model_name': model_name,
                'model_type': model_info.get('type', 'unknown') if model_name in self.loaded_models else 'unknown',
                'status': 'error',
                'error': str(e)
            }

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models"""
        return {
            name: {
                'type': info['type'],
                'input_features': info['input_features']
            }
            for name, info in self.loaded_models.items()
        }

    def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model"""
        if model_name not in self.loaded_models:
            return None

        info = self.loaded_models[model_name]
        return {
            'name': model_name,
            'type': info['type'],
            'input_features': info['input_features'],
            'loaded': True
        }
