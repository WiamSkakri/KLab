from app.main import app
import pytest
import json
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

# Import the FastAPI app
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


client = TestClient(app)


class TestHealthEndpoint:
    """Test health check functionality"""

    @patch('app.main.model_loader')
    def test_health_check_healthy(self, mock_loader):
        """Test health endpoint when system is healthy"""
        mock_loader.loaded_models = {"test_model": {"type": "test"}}

        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] == 1
        assert "system_info" in data
        assert "uptime_seconds" in data

    @patch('app.main.model_loader', None)
    def test_health_check_unhealthy(self):
        """Test health endpoint when model loader is not initialized"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["models_loaded"] == 0


class TestRootEndpoint:
    """Test root endpoint"""

    def test_root_endpoint(self):
        """Test the root endpoint returns API information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "CNN Execution Time Prediction API"
        assert data["version"] == "1.0.0"
        assert "docs" in data
        assert "health" in data


class TestModelsEndpoint:
    """Test model listing functionality"""

    @patch('app.main.model_loader')
    def test_get_models_success(self, mock_loader):
        """Test successful model listing"""
        mock_models = {
            "test_model_1": {"type": "xgboost", "input_features": ["batch_size"]},
            "test_model_2": {"type": "pytorch", "input_features": ["batch_size"]}
        }
        mock_loader.get_available_models.return_value = mock_models

        response = client.get("/models")
        assert response.status_code == 200

        data = response.json()
        assert data["total_models"] == 2
        assert "models" in data
        assert "test_model_1" in data["models"]

    @patch('app.main.model_loader', None)
    def test_get_models_no_loader(self):
        """Test model listing when loader is not initialized"""
        response = client.get("/models")
        assert response.status_code == 503
        assert "Model loader not initialized" in response.json()["detail"]


class TestPredictionEndpoint:
    """Test prediction functionality"""

    @patch('app.main.model_loader')
    def test_prediction_success(self, mock_loader):
        """Test successful prediction"""
        mock_loader.predict.return_value = {
            "prediction": 45.67,
            "model_name": "test_model",
            "model_type": "xgboost",
            "status": "success"
        }

        payload = {
            "model_name": "test_model",
            "cnn_config": {
                "batch_size": 32,
                "input_channels": 3,
                "input_height": 224,
                "input_width": 224,
                "output_channels": 64,
                "kernel_size": 3,
                "stride": 1
            }
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["execution_time_ms"] == 45.67
        assert data["model_name"] == "test_model"
        assert data["status"] == "success"

    @patch('app.main.model_loader')
    def test_prediction_invalid_model(self, mock_loader):
        """Test prediction with invalid model"""
        mock_loader.predict.side_effect = ValueError("Model not found")

        payload = {
            "model_name": "invalid_model",
            "cnn_config": {
                "batch_size": 32,
                "input_channels": 3,
                "input_height": 224,
                "input_width": 224,
                "output_channels": 64,
                "kernel_size": 3,
                "stride": 1
            }
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 400

    def test_prediction_invalid_input(self):
        """Test prediction with invalid input parameters"""
        payload = {
            "model_name": "test_model",
            "cnn_config": {
                "batch_size": -1,  # Invalid negative value
                "input_channels": 3,
                "input_height": 224,
                "input_width": 224,
                "output_channels": 64,
                "kernel_size": 3,
                "stride": 1
            }
        }

        response = client.post("/predict", json=payload)
        assert response.status_code == 422  # Validation error


class TestBenchmarkEndpoint:
    """Test benchmarking functionality"""

    @patch('app.main.model_loader')
    def test_benchmark_success(self, mock_loader):
        """Test successful model benchmarking"""
        mock_loader.loaded_models = {"test_model": {"type": "test"}}

        # Mock the prediction calls
        mock_loader.predict.return_value = {
            "prediction": 50.0,
            "status": "success"
        }

        response = client.get("/benchmark/test_model?iterations=5")
        assert response.status_code == 200

        data = response.json()
        assert "avg_latency_ms" in data
        assert "predictions_per_second" in data
        assert data["iterations"] == 5

    @patch('app.main.model_loader')
    def test_benchmark_model_not_found(self, mock_loader):
        """Test benchmarking non-existent model"""
        mock_loader.loaded_models = {}

        response = client.get("/benchmark/nonexistent_model")
        assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__])
