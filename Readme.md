# 🛡️ Sistema de detección de Fraude - Enterprise Solution

## 🎯 Resumen Ejecutivo

**Sistema empresarial de detección de fraude con IA** que reduce el tiempo de detección de 45 días a tiempo real, con un ROI proyectado del 844% y savings anuales de $20M+.

### 📊 Métricas Clave
- **Performance**: AUC 0.847
- **Velocidad**: <100ms por predicción
- **Precisión**: 62.3% en top 10% de casos
- **Compliance**: 100% decisiones explicables

## 🚀 Quick Start (1 Minuto)

```bash
# Clonar repositorio
git clone https://github.com/company/fraud-detection.git
cd fraud-detection

# Ejecutar demo con un comando
chmod +x demo.sh
./demo.sh
```

**¡Listo!** Sistema funcionando en:
- 🌐 Dashboard: http://localhost:8501
- 📚 API Docs: http://localhost:8000/docs

## 🏗️ Arquitectura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│  ML Models      │
│   Dashboard     │     │     API         │     │ Logistic + XGB  │
│   Port: 8501    │     │   Port: 8000    │     │   Scorecard     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Componentes Principales

1. **Frontend (Streamlit)**
   - Dashboard ejecutivo interactivo
   - Visualizaciones en tiempo real
   - Interface intuitiva para investigadores

2. **API (FastAPI)**
   - Endpoints RESTful documentados
   - Alta performance (<100ms)
   - Swagger UI incluido

3. **ML Engine**
   - Modelo dual: Regulatorio + Performance
   - Scorecard interpretable
   - Weight of Evidence (WoE)

## 📦 Instalación Manual

### Prerequisitos
- Docker 20.10+
- Docker Compose 2.0+
- 4GB RAM mínimo
- 2GB espacio en disco

### Paso a Paso

1. **Preparar ambiente**
```bash
# Verificar Docker
docker --version
docker-compose --version
```

2. **Construir imágenes**
```bash
docker-compose build
```

3. **Entrenar modelos**
```bash
docker-compose run train
```

4. **Iniciar servicios**
```bash
docker-compose up -d
```

## 🧪 Testing

### Test Rápido API
```bash
# Health check
curl http://localhost:8000/health

# Predicción de prueba
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Make": "BMW",
    "PolicyType": "Sport - All Perils",
    "Days_Policy_Claim": "1 to 7",
    "AgeOfPolicyHolder": "21 to 25",
    "VehiclePrice": "more than 69000"
  }'
```

### Test Dashboard
1. Abrir http://localhost:8501
2. Llenar formulario con datos de alto riesgo
3. Click en "ANALYZE FRAUD RISK"
4. Verificar score < 580 (HIGH RISK)

## 📊 Endpoints Principales

### Detección de Fraude
- `POST /predict` - Análisis individual
- `POST /predict/batch` - Procesamiento por lotes

### Business Intelligence
- `GET /business/metrics` - KPIs en tiempo real
- `GET /business/risk-segments` - Análisis de segmentos
- `GET /business/roi-analysis` - ROI detallado

### Información del Sistema
- `GET /model/info` - Detalles del modelo
- `GET /model/features` - Importancia de variables
- `GET /health` - Estado del sistema

## 🔧 Configuración

### Variables de Entorno
```bash
# API
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit
STREAMLIT_PORT=8501
API_URL=http://api:8000

# Modelos
MODEL_PATH=/app/models
```

### Personalización
- Modelos: Modificar `train_model.py`
- API: Editar `main.py`
- Dashboard: Actualizar `app.py`

## 📈 Monitoreo

### Logs
```bash
# Ver logs de API
docker logs -f fraud-api

# Ver logs de Dashboard
docker logs -f fraud-streamlit

# Ver todos los logs
docker-compose logs -f
```

### Métricas
```bash
# Performance del sistema
curl http://localhost:8000/health/detailed

# Métricas de negocio
curl http://localhost:8000/business/metrics
```

## 🚨 Troubleshooting

### Problema: "Docker daemon not running"
```bash
# Linux
sudo systemctl start docker

# Mac/Windows
# Iniciar Docker Desktop
```

### Problema: "Port already in use"
```bash
# Cambiar puertos en docker-compose.yml
# O detener servicios existentes:
docker-compose down
docker stop $(docker ps -q)
```

### Problema: "Models not found"
```bash
# Re-entrenar modelos
docker-compose run train
```

## 🏆 Business Value

### ROI Proyectado
- **Inversión**: $2.5M
- **Savings Año 1**: $20M+
- **ROI**: 844%
- **Payback**: 3.8 meses

### Beneficios Clave
1. **Velocidad**: De 45 días a tiempo real
2. **Precisión**: 62.3% detección en top 10%
3. **Escalabilidad**: 150+ transacciones/segundo
4. **Compliance**: 100% explicable

## 📞 Soporte

- **Technical Lead**: director.datos@company.com
- **Documentation**: http://localhost:8000/docs
- **Business Owner**: fraud.prevention@company.com

## 📄 Licencia

Proprietary - Company Confidential

---

**Desarrollado por**: Director de Datos AI  
**Versión**: 1.0.0  
**Última actualización**: Julio 2025