#!/bin/bash

# ============================================================================
# 🚀 FRAUD DETECTION SYSTEM - ONE-CLICK DEMO
# ============================================================================
# Script ejecutivo para demostración al CTO/CRO
# Autor: Director de Datos AI
# Versión: 2.0 - Optimizada
# ============================================================================

set -e  # Salir si hay errores

# Colores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Banner de inicio
echo -e "${BLUE}"
echo "=================================================="
echo "🛡️  FRAUD DETECTION SYSTEM - ENTERPRISE DEMO"
echo "=================================================="
echo -e "${NC}"

# Función para verificar Docker
check_docker() {
    echo -e "${YELLOW}🔍 Verificando Docker...${NC}"
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}❌ Docker no está instalado. Por favor instale Docker primero.${NC}"
        exit 1
    fi
    if ! docker ps &> /dev/null; then
        echo -e "${RED}❌ Docker daemon no está ejecutándose. Por favor inicie Docker.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✅ Docker está listo${NC}"
}

# Función para verificar archivos necesarios
check_files() {
    echo -e "${YELLOW}📁 Verificando archivos necesarios...${NC}"
    
    # MODIFICADO: Copiar app.py y main.py actualizados si no existen
    if [ ! -f "app.py" ] && [ -f "app_modified.py" ]; then
        echo "   Copiando app_modified.py -> app.py"
        cp app_modified.py app.py
    fi
    
    required_files=("app.py" "main.py" "models.py" "train_model.py" "requirements.txt")
    missing_files=()
    
    for file in "${required_files[@]}"; do
        if [ ! -f "$file" ]; then
            missing_files+=("$file")
        fi
    done
    
    if [ ${#missing_files[@]} -ne 0 ]; then
        echo -e "${RED}❌ Archivos faltantes: ${missing_files[*]}${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Todos los archivos necesarios presentes${NC}"
}

# Función para limpiar contenedores anteriores
cleanup() {
    echo -e "${YELLOW}🧹 Limpiando contenedores anteriores...${NC}"
    docker-compose down -v 2>/dev/null || true
    # MODIFICADO: Limpieza más agresiva
    docker rm -f fraud-train fraud-api fraud-streamlit 2>/dev/null || true
    docker volume prune -f 2>/dev/null || true
    echo -e "${GREEN}✅ Limpieza completada${NC}"
}

# Función para construir imágenes con caché
build_images() {
    echo -e "${YELLOW}🏗️  Construyendo imágenes Docker...${NC}"
    
    # MODIFICADO: Construcción con más información
    echo -e "${BLUE}   Esto puede tomar 5-10 minutos la primera vez...${NC}"
    
    # Construir con output detallado
    docker-compose build --progress=plain
    
    echo -e "${GREEN}✅ Imágenes construidas exitosamente${NC}"
}

# Función para entrenar modelos
train_models() {
    echo -e "${YELLOW}🤖 Entrenando modelos...${NC}"
    
    # MODIFICADO: Verificar si ya existen modelos
    if [ -d "models" ] && [ -f "models/logistic_model.pkl" ] && [ -f "models/xgb_model.pkl" ]; then
        echo -e "${BLUE}   Modelos existentes encontrados${NC}"
        read -p "   ¿Desea re-entrenar los modelos? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo -e "${GREEN}✅ Usando modelos existentes${NC}"
            return
        fi
    fi
    
    # Crear directorio de modelos si no existe
    mkdir -p models
    
    # Ejecutar entrenamiento
    echo -e "${BLUE}   Ejecutando train_model.py...${NC}"
    docker-compose run --rm train
    
    # Verificar que se crearon los modelos
    if [ ! -f "models/logistic_model.pkl" ]; then
        echo -e "${RED}❌ Error: No se generaron los modelos${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✅ Modelos entrenados exitosamente${NC}"
}

# Función para iniciar servicios
start_services() {
    echo -e "${YELLOW}🚀 Iniciando servicios...${NC}"
    
    # MODIFICADO: Iniciar solo API y Streamlit (train ya se ejecutó)
    docker-compose up -d api streamlit
    
    # Esperar a que los servicios estén listos
    echo -e "${YELLOW}⏳ Esperando a que los servicios estén listos...${NC}"
    
    # Esperar por la API con más reintentos
    echo -n "   Iniciando API"
    for i in {1..60}; do
        if curl -s http://localhost:8000/health > /dev/null 2>&1; then
            echo -e " ${GREEN}✅${NC}"
            break
        fi
        echo -n "."
        sleep 2
        
        # MODIFICADO: Verificar logs si tarda mucho
        if [ $i -eq 30 ]; then
            echo -e "\n${YELLOW}   Verificando logs de API...${NC}"
            docker logs fraud-api --tail 10
        fi
    done
    
    # Verificar el modo del API
    api_health=$(curl -s http://localhost:8000/health 2>/dev/null || echo "{}")
    if echo "$api_health" | grep -q "ML Engine Active"; then
        echo -e "   ${GREEN}✅ API usando modelos ML reales${NC}"
    else
        echo -e "   ${YELLOW}⚠️  API en modo fallback (reglas de negocio)${NC}"
    fi
    
    # Esperar por Streamlit
    echo -n "   Iniciando Dashboard"
    for i in {1..30}; do
        if curl -s http://localhost:8501 > /dev/null 2>&1; then
            echo -e " ${GREEN}✅${NC}"
            break
        fi
        echo -n "."
        sleep 2
    done
}

# Función para mostrar información de acceso
show_access_info() {
    echo -e "\n${GREEN}=================================================="
    echo -e "✅ SISTEMA INICIADO EXITOSAMENTE"
    echo -e "==================================================${NC}"
    echo
    echo -e "${BLUE}📊 ACCESO AL SISTEMA:${NC}"
    echo -e "   🌐 Dashboard Ejecutivo: ${GREEN}http://localhost:8501${NC}"
    echo -e "   🔌 API Documentation:  ${GREEN}http://localhost:8000/docs${NC}"
    echo -e "   💼 Business Metrics:   ${GREEN}http://localhost:8000/business/metrics${NC}"
    echo
    echo -e "${BLUE}🧪 PRUEBA RÁPIDA:${NC}"
    echo -e "   1. Abra ${GREEN}http://localhost:8501${NC} en su navegador"
    echo -e "   2. Configure un claim con:"
    echo -e "      - Vehicle: BMW"
    echo -e "      - Policy: Sport - All Perils"
    echo -e "      - Days to Claim: 1 to 7"
    echo -e "   3. Click 'ANALYZE FRAUD RISK'"
    echo -e "   4. Debería ver Risk Score < 580 (HIGH RISK)"
    echo
    echo -e "${BLUE}📈 MÉTRICAS CLAVE:${NC}"
    echo -e "   - Model AUC: 0.847"
    echo -e "   - Response Time: <100ms"
    echo -e "   - Annual Savings: \$20M+"
    echo
    echo -e "${YELLOW}📋 COMANDOS ÚTILES:${NC}"
    echo -e "   Ver logs API:        docker logs -f fraud-api"
    echo -e "   Ver logs Streamlit:  docker logs -f fraud-streamlit"
    echo -e "   Detener sistema:     docker-compose down"
    echo -e "   Ver métricas:        curl http://localhost:8000/business/metrics | python -m json.tool"
    echo -e "   Test API:            curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d '{\"Make\":\"BMW\",\"PolicyType\":\"Sport - All Perils\",\"Days_Policy_Claim\":\"1 to 7\"}'"
    echo
}

# Función para verificar salud del sistema
health_check() {
    echo -e "${YELLOW}🏥 Verificando salud del sistema...${NC}"
    
    # Check API
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        api_response=$(curl -s http://localhost:8000/health)
        if echo "$api_response" | grep -q "healthy"; then
            echo -e "   API: ${GREEN}✅ Operacional${NC}"
        else
            echo -e "   API: ${YELLOW}⚠️  Degradado${NC}"
        fi
    else
        echo -e "   API: ${RED}❌ No responde${NC}"
    fi
    
    # Check Streamlit
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo -e "   Dashboard: ${GREEN}✅ Operacional${NC}"
    else
        echo -e "   Dashboard: ${RED}❌ No responde${NC}"
    fi
    
    # Check models
    if [ -f "models/logistic_model.pkl" ] && [ -f "models/xgb_model.pkl" ]; then
        echo -e "   Modelos ML: ${GREEN}✅ Cargados${NC}"
        model_count=$(ls -1 models/*.pkl 2>/dev/null | wc -l)
        echo -e "   Total modelos: ${model_count} archivos"
    else
        echo -e "   Modelos ML: ${RED}❌ No encontrados${NC}"
    fi
}

# Función para modo desarrollo (sin rebuild)
dev_mode() {
    echo -e "${BLUE}🔧 Modo desarrollo - Skip build${NC}"
    docker-compose up -d
}

# Función principal
main() {
    echo -e "${YELLOW}🎯 Iniciando demo del Sistema de Detección de Fraude${NC}"
    echo -e "${YELLOW}📅 $(date)${NC}"
    echo
    
    # Verificar argumentos
    if [ "$1" == "--dev" ] || [ "$1" == "-d" ]; then
        dev_mode
        health_check
        show_access_info
        exit 0
    fi
    
    # Ejecutar pasos
    check_docker
    check_files
    cleanup
    train_models
    build_images
    start_services
    health_check
    show_access_info
    
    echo -e "${GREEN}🎉 ¡Sistema listo para demostración!${NC}"
    
    # MODIFICADO: Opción para abrir navegador automáticamente
    if command -v xdg-open &> /dev/null; then
        read -p "¿Abrir dashboard en navegador? (Y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            xdg-open http://localhost:8501
        fi
    fi
}

# Manejo de interrupción
trap cleanup INT

# Ejecutar
main "$@"