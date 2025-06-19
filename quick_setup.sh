#!/bin/bash

# Quick Setup Script for MLOps Pipeline
# This script automates the setup process after cloning the repository

echo "🚀 MLOps Pipeline Quick Setup"
echo "============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
print_status "Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_success "Docker and Docker Compose are installed"

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Are you in the correct directory?"
    exit 1
fi

print_success "Found docker-compose.yml"

# Step 1: Create required directories
print_status "Creating required directories..."
mkdir -p logs mlflow_artifacts
chmod 777 logs mlflow_artifacts
print_success "Directories created"

# Step 2: Start Docker containers
print_status "Starting Docker containers..."
docker-compose up -d

print_status "Waiting for containers to start (60 seconds)..."
sleep 60

# Step 3: Check container status
print_status "Checking container status..."
if ! docker-compose ps | grep -q "Up"; then
    print_error "Some containers failed to start. Check with: docker-compose logs"
    exit 1
fi
print_success "All containers are running"

# Step 4: Install Python packages
print_status "Installing Python packages in Airflow containers..."

containers=("airflow-scheduler" "airflow-webserver" "airflow-worker")
packages="mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib"

for container in "${containers[@]}"; do
    print_status "Installing packages in $container..."
    container_id=$(docker-compose ps -q $container)
    if [ -n "$container_id" ]; then
        docker exec $container_id pip3 install $packages
        if [ $? -eq 0 ]; then
            print_success "Packages installed in $container"
        else
            print_warning "Some packages may have failed to install in $container"
        fi
    else
        print_error "Container $container not found"
    fi
done

# Step 5: Create Airflow admin user
print_status "Creating Airflow admin user..."
webserver_id=$(docker-compose ps -q airflow-webserver)
docker exec $webserver_id airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin 2>/dev/null

print_success "Airflow admin user created (username: admin, password: admin)"

# Step 6: Copy source files to containers
print_status "Copying source files to Airflow containers..."
for container in "${containers[@]}"; do
    container_id=$(docker-compose ps -q $container)
    if [ -n "$container_id" ]; then
        print_status "Copying files to $container..."
        docker cp src/ $container_id:/opt/airflow/
        docker cp config/ $container_id:/opt/airflow/
        docker cp Dataset/ $container_id:/opt/airflow/
        print_success "Files copied to $container"
    fi
done

# Step 7: Set up directories in containers
print_status "Setting up directories in containers..."
for container in "${containers[@]}"; do
    container_id=$(docker-compose ps -q $container)
    if [ -n "$container_id" ]; then
        docker exec $container_id mkdir -p /opt/airflow/models /opt/airflow/mlflow_artifacts
        docker exec $container_id chmod 777 /opt/airflow/models /opt/airflow/mlflow_artifacts
    fi
done
print_success "Directories set up in all containers"

# Step 8: Wait for DAG registration
print_status "Waiting for DAG registration..."
sleep 30

# Step 9: Check DAG registration
print_status "Checking DAG registration..."
webserver_id=$(docker-compose ps -q airflow-webserver)
if docker exec $webserver_id airflow dags list 2>/dev/null | grep -q "lung_cancer_ml_pipeline"; then
    print_success "DAG 'lung_cancer_ml_pipeline' is registered"
else
    print_warning "DAG not yet visible. It may take a few more minutes."
fi

# Step 10: Final verification
print_status "Running final verification..."
./verify_setup.sh

echo
echo "================================"
print_success "Setup completed successfully! 🎉"
echo
echo "📌 Next Steps:"
echo "1. Open Airflow UI: http://localhost:8080"
echo "   - Username: admin"
echo "   - Password: admin"
echo
echo "2. Open MLflow UI: http://localhost:5000"
echo
echo "3. In Airflow UI:"
echo "   - Find the 'lung_cancer_ml_pipeline' DAG"
echo "   - Toggle it to ON (unpause)"
echo "   - Click the play button to trigger the pipeline"
echo
echo "4. Monitor the pipeline execution in the Airflow UI"
echo
echo "📋 Useful Commands:"
echo "   - Check container status: docker-compose ps"
echo "   - View logs: docker-compose logs <service-name>"
echo "   - Stop services: docker-compose down"
echo "   - Restart services: docker-compose restart"
echo
print_warning "If you encounter any issues, refer to the troubleshooting section in RUN.md"
