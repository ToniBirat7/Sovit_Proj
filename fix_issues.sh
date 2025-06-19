#!/bin/bash

# Quick Fix Script for Common MLOps Pipeline Issues
# This script addresses the most common issues encountered during setup

echo "🔧 MLOps Pipeline Quick Fix Tool"
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Are you in the correct directory?"
    exit 1
fi

echo "Select an issue to fix:"
echo "1. Fix Python package import issues in Airflow containers"
echo "2. Restart all services"
echo "3. Reinstall all Python packages"
echo "4. Fix DAG registration issues"
echo "5. Fix file permissions"
echo "6. Run full health check"
echo "0. Exit"

read -p "Enter your choice (0-6): " choice

case $choice in
    1)
        print_status "Fixing Python package import issues..."
        containers=("airflow-scheduler" "airflow-webserver" "airflow-worker")
        packages="mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib"
        
        for container in "${containers[@]}"; do
            container_id=$(docker-compose ps -q $container)
            if [ -n "$container_id" ]; then
                print_status "Checking $container..."
                if ! docker exec $container_id python3 -c "import mlflow, sklearn, redis, pandas, numpy" 2>/dev/null; then
                    print_warning "Fixing package imports in $container..."
                    docker exec $container_id pip3 install $packages --force-reinstall --no-deps
                    print_success "Fixed packages in $container"
                else
                    print_success "$container packages are working correctly"
                fi
            fi
        done
        ;;
        
    2)
        print_status "Restarting all services..."
        docker-compose restart
        print_status "Waiting for services to stabilize..."
        sleep 30
        print_success "Services restarted"
        ;;
        
    3)
        print_status "Reinstalling all Python packages..."
        containers=("airflow-scheduler" "airflow-webserver" "airflow-worker")
        packages="mlflow scikit-learn redis pandas numpy seaborn matplotlib joblib"
        
        for container in "${containers[@]}"; do
            container_id=$(docker-compose ps -q $container)
            if [ -n "$container_id" ]; then
                print_status "Reinstalling packages in $container..."
                docker exec $container_id pip3 install $packages --force-reinstall
            fi
        done
        print_success "All packages reinstalled"
        ;;
        
    4)
        print_status "Fixing DAG registration issues..."
        webserver_id=$(docker-compose ps -q airflow-webserver)
        scheduler_id=$(docker-compose ps -q airflow-scheduler)
        
        # Trigger DAG refresh
        if [ -n "$scheduler_id" ]; then
            docker exec $scheduler_id airflow dags reserialize
        fi
        
        print_status "Waiting for DAG registration..."
        sleep 15
        
        if [ -n "$webserver_id" ]; then
            if docker exec $webserver_id airflow dags list 2>/dev/null | grep -q "lung_cancer_ml_pipeline"; then
                print_success "DAG is now registered"
            else
                print_warning "DAG still not visible. Try waiting a few more minutes."
            fi
        fi
        ;;
        
    5)
        print_status "Fixing file permissions..."
        containers=("airflow-scheduler" "airflow-webserver" "airflow-worker")
        
        for container in "${containers[@]}"; do
            container_id=$(docker-compose ps -q $container)
            if [ -n "$container_id" ]; then
                print_status "Fixing permissions in $container..."
                docker exec $container_id chmod -R 755 /opt/airflow/src/
                docker exec $container_id chmod -R 777 /opt/airflow/models /opt/airflow/mlflow_artifacts
            fi
        done
        print_success "Permissions fixed"
        ;;
        
    6)
        print_status "Running full health check..."
        if [ -f "./verify_setup.sh" ]; then
            ./verify_setup.sh
        else
            print_error "verify_setup.sh not found"
        fi
        ;;
        
    0)
        print_status "Exiting..."
        exit 0
        ;;
        
    *)
        print_error "Invalid choice. Please select 0-6."
        exit 1
        ;;
esac

echo
print_success "Fix operation completed!"
echo
echo "📌 Useful Commands:"
echo "   - Check container status: docker-compose ps"
echo "   - View logs: docker-compose logs <service-name>"
echo "   - Access Airflow UI: http://localhost:8080"
echo "   - Access MLflow UI: http://localhost:5000"
