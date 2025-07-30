from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import uvicorn
import logging
import time
import psutil
from datetime import datetime
import json
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from models.model_loader import ModelLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    'ml_predictions_total', 'Total number of predictions', ['model_name', 'status'])
PREDICTION_LATENCY = Histogram(
    'ml_prediction_duration_seconds', 'Prediction latency', ['model_name'])
MODEL_LOAD_COUNTER = Counter(
    'ml_models_loaded_total', 'Total number of models loaded')

# Initialize FastAPI app
app = FastAPI(
    title="CNN Execution Time Prediction API",
    description="MLOps API for serving CNN execution time prediction models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model loader instance
model_loader = None

# Pydantic models for request/response


class CNNConfig(BaseModel):
    """CNN configuration for execution time prediction"""
    batch_size: int = Field(..., ge=1, le=1024,
                            description="Batch size for CNN")
    input_channels: int = Field(..., ge=1, le=1024,
                                description="Number of input channels")
    input_height: int = Field(..., ge=1, le=4096,
                              description="Input height in pixels")
    input_width: int = Field(..., ge=1, le=4096,
                             description="Input width in pixels")
    output_channels: int = Field(..., ge=1, le=2048,
                                 description="Number of output channels")
    kernel_size: int = Field(..., ge=1, le=64,
                             description="Convolution kernel size")
    stride: int = Field(..., ge=1, le=32, description="Convolution stride")


class PredictionRequest(BaseModel):
    """Request model for predictions"""
    model_name: str = Field(...,
                            description="Name of the model to use for prediction")
    cnn_config: CNNConfig = Field(...,
                                  description="CNN configuration parameters")


class PredictionResponse(BaseModel):
    """Response model for predictions"""
    execution_time_ms: float = Field(...,
                                     description="Predicted execution time in milliseconds")
    model_name: str = Field(..., description="Model used for prediction")
    model_type: str = Field(...,
                            description="Type of model (pytorch, xgboost, etc.)")
    prediction_timestamp: str = Field(...,
                                      description="ISO timestamp of prediction")
    status: str = Field(..., description="Prediction status")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    uptime_seconds: float
    models_loaded: int
    system_info: Dict[str, Any]


# Global variables for tracking
start_time = time.time()


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_loader
    logger.info("Starting CNN Execution Time Prediction API...")

    try:
        model_loader = ModelLoader()
        load_results = model_loader.load_all_available_models()

        successful_loads = sum(
            1 for success in load_results.values() if success)
        total_models = len(load_results)

        logger.info(
            f"Model loading complete: {successful_loads}/{total_models} models loaded successfully")

        # Update metrics
        MODEL_LOAD_COUNTER._value._value = successful_loads

        if successful_loads == 0:
            logger.warning("No models were loaded successfully!")
        else:
            logger.info("Available models:")
            for model_name, loaded in load_results.items():
                if loaded:
                    logger.info(f"  ✓ {model_name}")
                else:
                    logger.warning(f"  ✗ {model_name}")

    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        model_loader = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "CNN Execution Time Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check endpoint"""
    uptime = time.time() - start_time

    # System information
    system_info = {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "available_memory_gb": psutil.virtual_memory().available / (1024**3)
    }

    models_loaded = len(model_loader.loaded_models) if model_loader else 0

    return HealthResponse(
        status="healthy" if model_loader is not None else "unhealthy",
        timestamp=datetime.utcnow().isoformat(),
        uptime_seconds=uptime,
        models_loaded=models_loaded,
        system_info=system_info
    )


@app.get("/models", response_model=Dict[str, Any])
async def get_available_models():
    """Get information about all available models"""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Model loader not initialized")

    available_models = model_loader.get_available_models()

    return {
        "models": available_models,
        "total_models": len(available_models),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/models/{model_name}", response_model=Dict[str, Any])
async def get_model_info(model_name: str):
    """Get detailed information about a specific model"""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Model loader not initialized")

    model_info = model_loader.get_model_info(model_name)

    if model_info is None:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not found")

    return model_info


@app.post("/predict", response_model=PredictionResponse)
async def predict_execution_time(request: PredictionRequest):
    """Make prediction using specified model"""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Model loader not initialized")

    # Prepare input features
    input_features = {
        'batch_size': request.cnn_config.batch_size,
        'input_channels': request.cnn_config.input_channels,
        'input_height': request.cnn_config.input_height,
        'input_width': request.cnn_config.input_width,
        'output_channels': request.cnn_config.output_channels,
        'kernel_size': request.cnn_config.kernel_size,
        'stride': request.cnn_config.stride
    }

    # Measure prediction latency
    start_time_pred = time.time()

    try:
        # Make prediction
        result = model_loader.predict(request.model_name, input_features)

        if result['status'] == 'error':
            PREDICTION_COUNTER.labels(
                model_name=request.model_name, status='error').inc()
            raise HTTPException(
                status_code=400, detail=f"Prediction failed: {result['error']}")

        prediction_latency = time.time() - start_time_pred

        # Update metrics
        PREDICTION_COUNTER.labels(
            model_name=request.model_name, status='success').inc()
        PREDICTION_LATENCY.labels(
            model_name=request.model_name).observe(prediction_latency)

        return PredictionResponse(
            execution_time_ms=result['prediction'],
            model_name=result['model_name'],
            model_type=result['model_type'],
            prediction_timestamp=datetime.utcnow().isoformat(),
            status='success'
        )

    except ValueError as e:
        PREDICTION_COUNTER.labels(
            model_name=request.model_name, status='error').inc()
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        PREDICTION_COUNTER.labels(
            model_name=request.model_name, status='error').inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail="Internal server error during prediction")


@app.post("/predict/batch")
async def predict_batch(requests: List[PredictionRequest]):
    """Make batch predictions"""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Model loader not initialized")

    results = []

    for req in requests:
        try:
            prediction = await predict_execution_time(req)
            results.append(prediction.dict())
        except HTTPException as e:
            results.append({
                "model_name": req.model_name,
                "status": "error",
                "error": e.detail,
                "prediction_timestamp": datetime.utcnow().isoformat()
            })

    return {
        "results": results,
        "total_requests": len(requests),
        "successful_predictions": sum(1 for r in results if r.get("status") == "success"),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/benchmark/{model_name}")
async def benchmark_model(model_name: str, iterations: int = 100):
    """Benchmark a specific model performance"""
    if model_loader is None:
        raise HTTPException(
            status_code=503, detail="Model loader not initialized")

    if model_name not in model_loader.loaded_models:
        raise HTTPException(
            status_code=404, detail=f"Model '{model_name}' not found")

    # Sample CNN configuration for benchmarking
    sample_config = CNNConfig(
        batch_size=32,
        input_channels=3,
        input_height=224,
        input_width=224,
        output_channels=64,
        kernel_size=3,
        stride=1
    )

    latencies = []
    successful_predictions = 0

    for i in range(iterations):
        try:
            request = PredictionRequest(
                model_name=model_name, cnn_config=sample_config)
            start_time_bench = time.time()
            await predict_execution_time(request)
            latency = time.time() - start_time_bench
            latencies.append(latency)
            successful_predictions += 1
        except Exception as e:
            logger.warning(f"Benchmark iteration {i} failed: {str(e)}")

    if not latencies:
        raise HTTPException(
            status_code=500, detail="All benchmark iterations failed")

    return {
        "model_name": model_name,
        "iterations": iterations,
        "successful_predictions": successful_predictions,
        "avg_latency_ms": sum(latencies) / len(latencies) * 1000,
        "min_latency_ms": min(latencies) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "predictions_per_second": 1 / (sum(latencies) / len(latencies)),
        "timestamp": datetime.utcnow().isoformat()
    }

# Error handlers


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error",
                 "timestamp": datetime.utcnow().isoformat()}
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
