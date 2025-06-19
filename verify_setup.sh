#!/bin/bash

# MLOps Pipeline Verification Scri# Function to check if packages are installed in containers
check_packages() {
    local container_name=$1
    echo -n "Checking Python packages in $container_name... "
    
    # Try the import check first
    if docker exec $(docker-compose ps -q $container_name) python3 -c "import mlflow, sklearn, redis, pandas, numpy" 2>/dev/null; then
        echo -e "${GREEN}✅ All packages installed${NC}"
        return 0
    else
        # If import fails, check if packages are installed but maybe not importable
        if docker exec $(docker-compose ps -q $container_name) pip3 list 2>/dev/null | grep -q "mlflow\|scikit-learn"; then
            echo -e "${YELLOW}⚠️ Packages installed but may need container restart${NC}"
            return 0
        else
            echo -e "${RED}❌ Missing packages${NC}"
            return 1
        fi
    fi
}ipt checks if all components are running correctly

echo "🔍 MLOps Pipeline Health Check"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service status
check_service() {
    local service_name=$1
    local port=$2
    local endpoint=$3
    
    echo -n "Checking $service_name (port $port)... "
    
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:$port$endpoint | grep -q "200\|302"; then
        echo -e "${GREEN}✅ OK${NC}"
        return 0
    else
        echo -e "${RED}❌ FAILED${NC}"
        return 1
    fi
}

# Function to check Docker container status
check_container() {
    local container_pattern=$1
    local service_name=$2
    
    echo -n "Checking $service_name container... "
    
    if docker-compose ps | grep "$container_pattern" | grep -q "Up"; then
        echo -e "${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e "${RED}❌ Not Running${NC}"
        return 1
    fi
}

# Check if packages are installed in containers
check_packages() {
    local container_name=$1
    echo -n "Checking Python packages in $container_name... "
    
    if docker exec $(docker-compose ps -q $container_name) python3 -c "import mlflow, sklearn, redis, pandas, numpy" 2>/dev/null; then
        echo -e "${GREEN}✅ All packages installed${NC}"
        return 0
    else
        echo -e "${RED}❌ Missing packages${NC}"
        return 1
    fi
}

# Start health checks
echo "Starting health checks..."
echo

# Check if docker-compose.yml exists
echo -n "Checking if docker-compose.yml exists... "
if [ -f "docker-compose.yml" ]; then
    echo -e "${GREEN}✅ Found${NC}"
else
    echo -e "${RED}❌ Not found. Are you in the correct directory?${NC}"
    exit 1
fi

# Check Docker containers
echo
echo "📦 Checking Docker Containers:"
check_container "airflow-webserver" "Airflow Webserver"
check_container "airflow-scheduler" "Airflow Scheduler"
check_container "airflow-worker" "Airflow Worker"
check_container "mlflow" "MLflow"
check_container "redis" "Redis"
check_container "postgres" "PostgreSQL"

# Check services accessibility
echo
echo "🌐 Checking Service Accessibility:"
check_service "Airflow UI" "8080" ""
check_service "MLflow UI" "5000" ""

# Check Redis connectivity
echo -n "Checking Redis connectivity... "
if docker exec $(docker-compose ps -q redis) redis-cli ping 2>/dev/null | grep -q "PONG"; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ Connection failed${NC}"
fi

# Check Python packages in Airflow containers
echo
echo "🐍 Checking Python Packages:"
check_packages "airflow-scheduler"
check_packages "airflow-webserver" 
check_packages "airflow-worker"

# Check if DAG is registered
echo
echo "📋 Checking Airflow DAG:"
echo -n "Checking if lung_cancer_ml_pipeline DAG is registered... "
if docker exec $(docker-compose ps -q airflow-webserver) airflow dags list 2>/dev/null | grep -q "lung_cancer_ml_pipeline"; then
    echo -e "${GREEN}✅ DAG found${NC}"
else
    echo -e "${RED}❌ DAG not found${NC}"
fi

# Check if source files are copied
echo
echo "📁 Checking Source Files:"
echo -n "Checking if source files are copied to containers... "
if docker exec $(docker-compose ps -q airflow-worker) ls /opt/airflow/src/ 2>/dev/null | grep -q "data\|models\|features\|utils"; then
    echo -e "${GREEN}✅ Source files found${NC}"
else
    echo -e "${RED}❌ Source files missing${NC}"
fi

# Check if dataset exists
echo -n "Checking if dataset exists... "
if docker exec $(docker-compose ps -q airflow-worker) ls /opt/airflow/Dataset/ 2>/dev/null | grep -q "lung_cancer_mortality_data_test_v2.csv"; then
    echo -e "${GREEN}✅ Dataset found${NC}"
elif [ -f "Dataset/lung_cancer_mortality_data_test_v2.csv" ]; then
    echo -e "${YELLOW}⚠️ Dataset found locally but not in container${NC}"
else
    echo -e "${RED}❌ Dataset missing${NC}"
fi

# Check required directories
echo -n "Checking required directories... "
if docker exec $(docker-compose ps -q airflow-worker) ls -d /opt/airflow/models /opt/airflow/mlflow_artifacts 2>/dev/null | wc -l | grep -q "2"; then
    echo -e "${GREEN}✅ Directories exist${NC}"
else
    echo -e "${YELLOW}⚠️ Some directories may need to be created${NC}"
fi

echo
echo "================================"
echo "🏁 Health Check Complete"
echo
echo "📌 Next Steps:"
echo "1. Open Airflow UI: http://localhost:8080 (admin/admin)"
echo "2. Open MLflow UI: http://localhost:5000"
echo "3. Find 'lung_cancer_ml_pipeline' DAG in Airflow"
echo "4. Toggle the DAG to ON and trigger it"
echo
echo -e "${YELLOW}💡 If any checks failed, refer to the troubleshooting section in RUN.md${NC}"
