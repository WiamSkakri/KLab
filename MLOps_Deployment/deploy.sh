#!/bin/bash

# CNN Execution Time Prediction - MLOps Deployment Script
# This script handles the complete deployment of the ML service

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="cnn-execution-predictor"
DOCKER_IMAGE_NAME="cnn-predictor-api"
STREAMLIT_IMAGE_NAME="cnn-predictor-frontend"
VERSION=${1:-"latest"}

echo -e "${BLUE}🚀 Starting MLOps Deployment for CNN Execution Time Predictor${NC}"
echo -e "${BLUE}=================================================${NC}"

# Function to print status
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    echo -e "${BLUE}🔍 Checking prerequisites...${NC}"
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    # Check if Docker Compose is installed
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    print_status "All prerequisites met"
}

# Create necessary directories
setup_directories() {
    echo -e "${BLUE}📁 Setting up directories...${NC}"
    
    mkdir -p logs
    mkdir -p data
    mkdir -p monitoring/grafana-provisioning/dashboards
    mkdir -p monitoring/grafana-provisioning/datasources
    mkdir -p nginx/ssl
    
    print_status "Directories created"
}

# Create Grafana configuration
setup_grafana() {
    echo -e "${BLUE}📊 Setting up Grafana configuration...${NC}"
    
    # Create datasource configuration
    cat > monitoring/grafana-provisioning/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create dashboard configuration
    cat > monitoring/grafana-provisioning/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    # Create a sample dashboard for ML metrics
    cat > monitoring/grafana-provisioning/dashboards/ml-metrics.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "ML API Metrics",
    "tags": ["ml", "api"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Prediction Requests",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ml_predictions_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Prediction Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, ml_prediction_duration_seconds_bucket)",
            "legendFormat": "95th percentile"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "5s"
  }
}
EOF
    
    print_status "Grafana configuration created"
}

# Create Nginx configuration
setup_nginx() {
    echo -e "${BLUE}🌐 Setting up Nginx configuration...${NC}"
    
    cat > nginx/nginx.conf << EOF
events {
    worker_connections 1024;
}

http {
    upstream ml_api {
        server ml-api:8000;
    }
    
    upstream frontend {
        server streamlit:8501;
    }
    
    upstream grafana {
        server grafana:3000;
    }

    server {
        listen 80;
        server_name localhost;

        # ML API
        location /api/ {
            rewrite ^/api/(.*)$ /\$1 break;
            proxy_pass http://ml_api;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
            proxy_http_version 1.1;
            proxy_set_header Upgrade \$http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Grafana
        location /grafana/ {
            rewrite ^/grafana/(.*)$ /\$1 break;
            proxy_pass http://grafana;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto \$scheme;
        }
    }
}
EOF
    
    print_status "Nginx configuration created"
}

# Create Streamlit Dockerfile
create_streamlit_dockerfile() {
    echo -e "${BLUE}🎨 Creating Streamlit Dockerfile...${NC}"
    
    cat > Dockerfile.frontend << EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir streamlit

COPY app/frontend/ ./

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
EOF
    
    print_status "Streamlit Dockerfile created"
}

# Build Docker images
build_images() {
    echo -e "${BLUE}🐳 Building Docker images...${NC}"
    
    # Build main API image
    echo "Building main API image..."
    docker build -t ${DOCKER_IMAGE_NAME}:${VERSION} .
    
    # Build Streamlit frontend image
    echo "Building Streamlit frontend image..."
    docker build -f Dockerfile.frontend -t ${STREAMLIT_IMAGE_NAME}:${VERSION} .
    
    print_status "Docker images built successfully"
}

# Update docker-compose with Streamlit service
update_docker_compose() {
    echo -e "${BLUE}📝 Updating docker-compose configuration...${NC}"
    
    # Add Streamlit service to docker-compose.yml
    cat >> docker-compose.yml << EOF

  # Streamlit Frontend
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    ports:
      - "8501:8501"
    environment:
      - API_BASE_URL=http://ml-api:8000
    depends_on:
      - ml-api
    networks:
      - ml-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
    
    print_status "Docker compose updated"
}

# Deploy the stack
deploy_stack() {
    echo -e "${BLUE}🚀 Deploying the complete stack...${NC}"
    
    # Stop any existing containers
    docker-compose down
    
    # Start the services
    docker-compose up -d --build
    
    print_status "Stack deployed successfully"
}

# Health check
health_check() {
    echo -e "${BLUE}🏥 Running health checks...${NC}"
    
    # Wait for services to start
    sleep 30
    
    # Check API health
    echo "Checking API health..."
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_status "API is healthy"
    else
        print_warning "API health check failed"
    fi
    
    # Check Streamlit
    echo "Checking Streamlit frontend..."
    if curl -f http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        print_status "Streamlit frontend is healthy"
    else
        print_warning "Streamlit health check failed"
    fi
    
    # Check Prometheus
    echo "Checking Prometheus..."
    if curl -f http://localhost:9090/-/healthy > /dev/null 2>&1; then
        print_status "Prometheus is healthy"
    else
        print_warning "Prometheus health check failed"
    fi
    
    # Check Grafana
    echo "Checking Grafana..."
    if curl -f http://localhost:3000/api/health > /dev/null 2>&1; then
        print_status "Grafana is healthy"
    else
        print_warning "Grafana health check failed"
    fi
}

# Show deployment information
show_deployment_info() {
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo -e "${BLUE}=================================================${NC}"
    echo -e "${YELLOW}📋 Service URLs:${NC}"
    echo -e "   🔗 Main Application: http://localhost:80"
    echo -e "   🤖 API Documentation: http://localhost:8000/docs"
    echo -e "   📊 Grafana Dashboard: http://localhost:3000 (admin/admin123)"
    echo -e "   📈 Prometheus Metrics: http://localhost:9090"
    echo -e "   💻 Streamlit UI: http://localhost:8501"
    echo -e ""
    echo -e "${YELLOW}🛠️  Management Commands:${NC}"
    echo -e "   View logs: docker-compose logs -f"
    echo -e "   Stop services: docker-compose down"
    echo -e "   Restart services: docker-compose restart"
    echo -e "   Scale API: docker-compose up -d --scale ml-api=3"
    echo -e ""
    echo -e "${YELLOW}📁 Important directories:${NC}"
    echo -e "   📊 Logs: ./logs/"
    echo -e "   💾 Data: ./data/"
    echo -e "   🔧 Monitoring config: ./monitoring/"
}

# Main deployment flow
main() {
    check_prerequisites
    setup_directories
    setup_grafana
    setup_nginx
    create_streamlit_dockerfile
    build_images
    update_docker_compose
    deploy_stack
    health_check
    show_deployment_info
}

# Run main function
main "$@" 