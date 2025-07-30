# CNN Execution Time Predictor - MLOps Deployment

A production-ready MLOps system for serving CNN execution time prediction models with comprehensive monitoring, scaling, and user interfaces.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend UI   │    │   Nginx Proxy   │    │   ML API        │
│   (Streamlit)   │────│   Load Balancer │────│   (FastAPI)     │
│   Port: 8501    │    │   Port: 80      │    │   Port: 8000    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                       │
                                │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   Prometheus    │    │   Redis Cache   │
│   Monitoring    │────│   Metrics       │    │   Session Store │
│   Port: 3000    │    │   Port: 9090    │    │   Port: 6379    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker (>= 20.10)
- Docker Compose (>= 1.29)
- 8GB+ RAM recommended
- 10GB+ free disk space

### One-Command Deployment

```bash
cd MLOps_Deployment
chmod +x deploy.sh
./deploy.sh
```

### Manual Deployment

```bash
# 1. Build and start services
docker-compose up -d --build

# 2. Check health
curl http://localhost:8000/health

# 3. Access the applications
open http://localhost        # Main UI
open http://localhost:8000/docs  # API Documentation
open http://localhost:3000   # Grafana Dashboard (admin/admin123)
```

## 📊 Features

### 🔮 Model Serving
- **Multiple Model Support**: Neural Networks, XGBoost, Random Forest, Polynomial Regression
- **Automatic Model Discovery**: Scans and loads all available trained models
- **Real-time Predictions**: Sub-second prediction latency
- **Batch Processing**: Handle multiple predictions efficiently
- **Model Benchmarking**: Performance testing and comparison

### 🌐 User Interfaces
- **Interactive Web UI**: Streamlit-based frontend with model comparison
- **REST API**: Full OpenAPI/Swagger documentation
- **Parameter Presets**: Quick configurations for popular CNN architectures
- **Real-time Visualization**: Charts and gauges for prediction results

### 📈 Monitoring & Observability
- **Prometheus Metrics**: Request rates, latency, model performance
- **Grafana Dashboards**: Visual monitoring with alerts
- **Health Checks**: Automated service health monitoring
- **System Metrics**: CPU, Memory, Disk usage tracking
- **Application Logs**: Structured logging with rotation

### 🔧 Production Features
- **Auto-scaling**: Docker Compose scaling capabilities
- **Load Balancing**: Nginx reverse proxy with health checks
- **Security**: Non-root containers, input validation
- **Caching**: Redis for session storage and caching
- **Error Handling**: Graceful degradation and error recovery

## 🎯 API Endpoints

### Core Prediction API

```http
POST /predict
Content-Type: application/json

{
  "model_name": "xgb_prediction-training-gpul40s",
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
```

**Response:**
```json
{
  "execution_time_ms": 45.67,
  "model_name": "xgb_prediction-training-gpul40s",
  "model_type": "xgboost",
  "prediction_timestamp": "2024-01-15T10:30:00Z",
  "status": "success"
}
```

### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health status |
| `/models` | GET | List available models |
| `/models/{name}` | GET | Model details |
| `/predict/batch` | POST | Batch predictions |
| `/benchmark/{model}` | GET | Model benchmarking |
| `/metrics` | GET | Prometheus metrics |
| `/docs` | GET | API documentation |

## 🎛️ Configuration

### Environment Variables

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO

# Redis Configuration
REDIS_URL=redis://redis:6379

# Model Configuration
MODELS_DIR=/app/CNN_Algorithms
MODEL_CACHE_SIZE=100MB
```

### Model Configuration

The system automatically discovers models in the following formats:

```
CNN_Algorithms/
├── prediction-training-cpu/
│   ├── nn/output/best_model.pth
│   ├── xgboost/best_xgb_model.joblib
│   └── rdf/best_rf_model.joblib
├── prediction-training-gpul40s/
│   └── ...
└── polynomial-regression/
    └── best_lasso_model.joblib
```

## 📊 Monitoring & Metrics

### Key Metrics Tracked

1. **Prediction Metrics**
   - Request rate (requests/second)
   - Prediction latency (p95, p99)
   - Model accuracy metrics
   - Error rates by model

2. **System Metrics**
   - CPU and memory usage
   - Disk I/O and space
   - Network throughput
   - Service uptime

3. **Business Metrics**
   - Popular models usage
   - CNN configuration patterns
   - User engagement

### Grafana Dashboards

Access Grafana at `http://localhost:3000` (admin/admin123)

Pre-configured dashboards:
- **ML API Overview**: Request rates, latency, errors
- **Model Performance**: Individual model metrics
- **System Health**: Infrastructure monitoring
- **User Analytics**: Usage patterns and trends

## 🔧 Operations

### Scaling

```bash
# Scale API instances
docker-compose up -d --scale ml-api=3

# Scale with resource limits
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Monitoring

```bash
# View logs
docker-compose logs -f ml-api

# Check metrics
curl http://localhost:9090/metrics

# Health checks
curl http://localhost:8000/health
```

### Backup & Recovery

```bash
# Backup models and data
tar -czf backup-$(date +%Y%m%d).tar.gz data/ CNN_Algorithms/

# Restore from backup
tar -xzf backup-20240115.tar.gz
```

## 🚀 Cloud Deployment

### AWS ECS Deployment

```bash
# Build and push to ECR
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-west-2.amazonaws.com
docker build -t cnn-predictor .
docker tag cnn-predictor:latest 123456789012.dkr.ecr.us-west-2.amazonaws.com/cnn-predictor:latest
docker push 123456789012.dkr.ecr.us-west-2.amazonaws.com/cnn-predictor:latest

# Deploy to ECS
aws ecs update-service --cluster ml-cluster --service cnn-predictor --force-new-deployment
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cnn-predictor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cnn-predictor
  template:
    metadata:
      labels:
        app: cnn-predictor
    spec:
      containers:
      - name: api
        image: cnn-predictor:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

## 🧪 Testing

### API Testing

```bash
# Health check
curl -f http://localhost:8000/health

# Model prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "xgb_prediction-training-gpul40s",
    "cnn_config": {
      "batch_size": 32,
      "input_channels": 3,
      "input_height": 224,
      "input_width": 224,
      "output_channels": 64,
      "kernel_size": 3,
      "stride": 1
    }
  }'

# Load testing with apache bench
ab -n 1000 -c 10 -T 'application/json' \
   -p test_payload.json \
   http://localhost:8000/predict
```

### Performance Benchmarking

```bash
# Benchmark specific model
curl "http://localhost:8000/benchmark/xgb_prediction-training-gpul40s?iterations=100"
```

## 🔐 Security

### Production Security Checklist

- [ ] Change default Grafana password
- [ ] Configure SSL/TLS certificates
- [ ] Set up API rate limiting
- [ ] Configure firewall rules
- [ ] Enable authentication for API endpoints
- [ ] Regular security updates
- [ ] Monitor for unusual access patterns

### API Security

```python
# Add API key authentication (example)
from fastapi import Depends, HTTPException, security

api_key_header = security.APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key
```

## 🛟 Troubleshooting

### Common Issues

1. **Models not loading**
   ```bash
   # Check model paths
   docker-compose exec ml-api ls -la /app/CNN_Algorithms/
   
   # Check logs
   docker-compose logs ml-api
   ```

2. **High memory usage**
   ```bash
   # Monitor memory
   docker stats
   
   # Reduce model cache size
   export MODEL_CACHE_SIZE=50MB
   ```

3. **Slow predictions**
   ```bash
   # Check system resources
   docker-compose exec ml-api top
   
   # Monitor API metrics
   curl http://localhost:9090/graph?g0.expr=ml_prediction_duration_seconds
   ```

### Log Analysis

```bash
# Search for errors
docker-compose logs ml-api | grep ERROR

# Monitor prediction latency
docker-compose logs ml-api | grep "prediction_latency"

# Check model loading
docker-compose logs ml-api | grep "loaded successfully"
```

## 📚 Model Management

### Adding New Models

1. **Train and save models** in the `CNN_Algorithms` directory
2. **Follow naming conventions**: `best_{model_type}_model.{pth|joblib}`
3. **Restart services**: `docker-compose restart ml-api`
4. **Verify loading**: Check `/models` endpoint

### Model Versioning

```bash
# Save model with version
cp best_model.pth models/v1.2.0/best_model.pth

# Update model loader to use versioned models
# Implement A/B testing between model versions
```

## 🔄 CI/CD Pipeline

### GitHub Actions (Example)

```yaml
# .github/workflows/deploy.yml
name: Deploy ML API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Build and test
      run: |
        docker build -t cnn-predictor .
        docker run --rm cnn-predictor python -m pytest
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        ./deploy.sh production
```

## 📈 Performance Optimization

### Model Optimization
- **Model quantization** for smaller memory footprint
- **Model pruning** to reduce inference time
- **Batch prediction optimization**
- **Model caching** strategies

### Infrastructure Optimization
- **Horizontal scaling** with load balancers
- **GPU acceleration** for neural network models
- **CDN caching** for static assets
- **Database connection pooling**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd MLOps_Deployment

# Set up development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run in development mode
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and API docs
- **Issues**: Submit GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team

---

## 🎉 Success Metrics

After deployment, you should see:

- ✅ **Sub-second prediction latency** for most models
- ✅ **99%+ uptime** with health checks
- ✅ **Auto-scaling** based on load
- ✅ **Comprehensive monitoring** with alerts
- ✅ **User-friendly interface** for easy interaction

Your CNN execution time prediction system is now production-ready! 🚀 