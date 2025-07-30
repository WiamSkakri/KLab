import pytest
import requests
import time
import json
import os

# Integration tests that run against the actual deployed system


class TestIntegration:
    """Integration tests for the complete MLOps system"""

    BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    def test_api_health_check(self):
        """Test that the API is responding and healthy"""
        response = requests.get(f"{self.BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert data["models_loaded"] > 0
        assert "system_info" in data

    def test_models_discovery(self):
        """Test that models are discovered and available"""
        response = requests.get(f"{self.BASE_URL}/models")
        assert response.status_code == 200

        data = response.json()
        assert data["total_models"] > 0
        assert "models" in data

        # Verify model structure
        for model_name, model_info in data["models"].items():
            assert "type" in model_info
            assert "input_features" in model_info
            assert len(model_info["input_features"]) > 0

    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        # Get available models
        models_response = requests.get(f"{self.BASE_URL}/models")
        assert models_response.status_code == 200

        models_data = models_response.json()
        assert models_data["total_models"] > 0

        # Use the first available model
        model_name = list(models_data["models"].keys())[0]

        # Make a prediction
        prediction_payload = {
            "model_name": model_name,
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

        prediction_response = requests.post(
            f"{self.BASE_URL}/predict",
            json=prediction_payload
        )
        assert prediction_response.status_code == 200

        prediction_data = prediction_response.json()
        assert prediction_data["status"] == "success"
        assert prediction_data["execution_time_ms"] > 0
        assert prediction_data["model_name"] == model_name
        assert "prediction_timestamp" in prediction_data

    def test_batch_prediction(self):
        """Test batch prediction functionality"""
        # Get available models
        models_response = requests.get(f"{self.BASE_URL}/models")
        models_data = models_response.json()
        model_name = list(models_data["models"].keys())[0]

        # Create batch requests
        batch_requests = [
            {
                "model_name": model_name,
                "cnn_config": {
                    "batch_size": 16,
                    "input_channels": 3,
                    "input_height": 224,
                    "input_width": 224,
                    "output_channels": 32,
                    "kernel_size": 3,
                    "stride": 1
                }
            },
            {
                "model_name": model_name,
                "cnn_config": {
                    "batch_size": 64,
                    "input_channels": 3,
                    "input_height": 512,
                    "input_width": 512,
                    "output_channels": 128,
                    "kernel_size": 5,
                    "stride": 2
                }
            }
        ]

        response = requests.post(
            f"{self.BASE_URL}/predict/batch", json=batch_requests)
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 2
        assert data["successful_predictions"] > 0
        assert len(data["results"]) == 2

    def test_model_benchmarking(self):
        """Test model benchmarking functionality"""
        # Get available models
        models_response = requests.get(f"{self.BASE_URL}/models")
        models_data = models_response.json()
        model_name = list(models_data["models"].keys())[0]

        # Run benchmark with small number of iterations for speed
        response = requests.get(
            f"{self.BASE_URL}/benchmark/{model_name}?iterations=10")
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == model_name
        assert data["iterations"] == 10
        assert data["successful_predictions"] > 0
        assert data["avg_latency_ms"] > 0
        assert data["predictions_per_second"] > 0

    def test_api_documentation(self):
        """Test that API documentation is accessible"""
        response = requests.get(f"{self.BASE_URL}/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_prometheus_metrics(self):
        """Test that Prometheus metrics are available"""
        response = requests.get(f"{self.BASE_URL}/metrics")
        assert response.status_code == 200

        metrics_text = response.text
        # Check for key metrics
        assert "ml_predictions_total" in metrics_text
        assert "ml_prediction_duration_seconds" in metrics_text
        assert "ml_models_loaded_total" in metrics_text

    def test_error_handling(self):
        """Test API error handling"""
        # Test with invalid model name
        invalid_payload = {
            "model_name": "nonexistent_model",
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

        response = requests.post(
            f"{self.BASE_URL}/predict", json=invalid_payload)
        assert response.status_code == 400

        # Test with invalid input parameters
        invalid_params_payload = {
            "model_name": "any_model",
            "cnn_config": {
                "batch_size": -1,  # Invalid
                "input_channels": 0,  # Invalid
                "input_height": 224,
                "input_width": 224,
                "output_channels": 64,
                "kernel_size": 3,
                "stride": 1
            }
        }

        response = requests.post(
            f"{self.BASE_URL}/predict", json=invalid_params_payload)
        assert response.status_code == 422

    def test_performance_requirements(self):
        """Test that the system meets performance requirements"""
        # Get available models
        models_response = requests.get(f"{self.BASE_URL}/models")
        models_data = models_response.json()
        model_name = list(models_data["models"].keys())[0]

        # Test prediction latency
        prediction_payload = {
            "model_name": model_name,
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

        start_time = time.time()
        response = requests.post(
            f"{self.BASE_URL}/predict", json=prediction_payload)
        end_time = time.time()

        assert response.status_code == 200

        # Should respond within 5 seconds (adjust as needed)
        latency = end_time - start_time
        assert latency < 5.0, f"Prediction took {latency:.2f}s, expected < 5.0s"

        # Test concurrent requests
        import concurrent.futures

        def make_request():
            return requests.post(f"{self.BASE_URL}/predict", json=prediction_payload)

        # Test 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(5)]
            results = [future.result()
                       for future in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        for result in results:
            assert result.status_code == 200


class TestSystemResilience:
    """Test system resilience and recovery"""

    BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

    def test_high_load_handling(self):
        """Test system behavior under high load"""
        # Get available models
        models_response = requests.get(f"{self.BASE_URL}/models")
        models_data = models_response.json()
        model_name = list(models_data["models"].keys())[0]

        prediction_payload = {
            "model_name": model_name,
            "cnn_config": {
                "batch_size": 1,
                "input_channels": 3,
                "input_height": 32,
                "input_width": 32,
                "output_channels": 16,
                "kernel_size": 3,
                "stride": 1
            }
        }

        # Send many requests quickly
        successful_requests = 0
        total_requests = 50

        for _ in range(total_requests):
            try:
                response = requests.post(
                    f"{self.BASE_URL}/predict",
                    json=prediction_payload,
                    timeout=10
                )
                if response.status_code == 200:
                    successful_requests += 1
            except requests.RequestException:
                pass  # Expected under high load

        # Should handle at least 80% of requests successfully
        success_rate = successful_requests / total_requests
        assert success_rate >= 0.8, f"Success rate {success_rate:.2%} < 80%"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
