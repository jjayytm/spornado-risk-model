# Deployment Guide

## Production Deployment

### Prerequisites

- Python 3.8+
- Virtual environment set up
- Dependencies installed: `pip install -r requirements.txt`
- Models trained: `models/model_performance_results.csv`

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/spornado-disease-prediction.git
cd spornado-disease-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configuration

1. Edit `config/config.yaml` for your environment
2. Set environment variables:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL=INFO
export MODEL_PATH=models/
export DATA_PATH=data/processed/
```

### Step 3: Pre-Deployment Testing

```bash
# Test data pipeline
python -m pytest tests/test_models.py -v

# Test model predictions
python src/build_crop_models.py --data data/raw/spornado_weather_spore_data.csv

# Verify model performance
cat models/model_performance_results.csv
```

Expected output:
```
Crop,Train_AUC,Test_AUC,Test_F1,Model_Type
Corn,0.9938,0.9967,0.9835,Logistic Regression
Soybean,0.9980,1.0000,1.0000,Logistic Regression
Potato,0.8950,0.9450,0.9125,Logistic Regression
```

### Step 4: Deploy Models to Production

#### Option A: Manual Deployment

```bash
# Create production directory
mkdir -p /var/www/spornado-models
cp -r models/ /var/www/spornado-models/
cp -r data/processed/ /var/www/spornado-data/
cp config/config.yaml /var/www/spornado-config/
```

#### Option B: Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ src/
COPY models/ models/
COPY data/ data/
COPY config/ config/

# Set environment
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=INFO

# Expose API port
EXPOSE 5000

# Run application
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
```

Build and run:

```bash
docker build -t spornado-disease-prediction:1.0 .
docker run -p 5000:5000 spornado-disease-prediction:1.0
```

#### Option C: Cloud Deployment (AWS/GCP/Azure)

```bash
# Package application
zip -r spornado-model.zip src/ models/ data/ config/ requirements.txt

# Upload to cloud (example: AWS Lambda)
aws lambda create-function \
  --function-name spornado-disease-prediction \
  --runtime python3.9 \
  --role arn:aws:iam::ACCOUNT_ID:role/lambda-role \
  --handler src.api:handler \
  --zip-file fileb://spornado-model.zip
```

### Step 5: API Setup (Optional - Flask)

Create `src/api.py`:

```python
from flask import Flask, request, jsonify
from src.build_crop_models import DiseaseRiskModel
import logging

app = Flask(__name__)
model = DiseaseRiskModel()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict disease risk
    
    Request JSON:
    {
        "crop": "Corn",
        "spore_count": 150,
        "temp_max": 28,
        "temp_min": 18,
        "humidity": 85,
        ...
    }
    """
    try:
        data = request.json
        risk_score = model.predict(
            crop=data.get('crop'),
            spore_count=data.get('spore_count'),
            # ... other features
        )
        return jsonify({
            'status': 'success',
            'risk_score': risk_score,
            'recommendation': 'Plant resistant variety'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

Run:
```bash
python src/api.py
```

### Step 6: Monitoring & Logging

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/production.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info('Model deployed successfully')
```

Monitor logs:
```bash
tail -f logs/production.log
```

### Step 7: Model Retraining Pipeline

When new farmer variety data becomes available:

```bash
# Step 1: Integrate variety data
python src/integrate_variety_data.py \
  --input data/raw/farmer_variety_records.csv \
  --output data/processed/spornado_with_variety_features.csv

# Step 2: Retrain models
python src/build_crop_models_with_variety.py \
  --data data/processed/spornado_with_variety_features.csv

# Step 3: Validate new models
python -m pytest tests/test_models.py -v

# Step 4: Deploy if validated
cp models/model_performance_results_with_variety.csv models/model_performance_results.csv
systemctl restart spornado-service  # If using systemd
```

### Step 8: Health Checks

```bash
# Check model files exist
ls -la models/model_performance_results.csv

# Verify data files
ls -la data/processed/

# Test API endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "crop": "Corn",
    "spore_count": 150,
    "temp_max": 28,
    "temp_min": 18,
    "humidity": 85
  }'
```

---

## Scaling Considerations

### Load Balancing
For high-volume predictions, use nginx:

```nginx
upstream spornado {
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
    server 127.0.0.1:5003;
}

server {
    listen 80;
    location /predict {
        proxy_pass http://spornado;
    }
}
```

### Caching
```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_disease_tolerance(variety_name):
    # Cached lookups for variety data
    pass
```

### Database Integration
For large-scale deployments:

```python
import psycopg2

conn = psycopg2.connect("dbname=spornado user=admin")
cursor = conn.cursor()

# Store predictions
cursor.execute("""
    INSERT INTO predictions (location_id, crop, risk_score, timestamp)
    VALUES (%s, %s, %s, %s)
""", (location_id, crop, risk_score, timestamp))
```

---

## Troubleshooting

### Issue: Model not loading
```bash
# Check file exists
file models/model_performance_results.csv

# Check permissions
ls -la models/

# Verify format
head models/model_performance_results.csv
```

### Issue: API timeout
- Increase timeout in config: `api.timeout: 30`
- Implement caching for predictions
- Use async processing for batch predictions

### Issue: Memory issues
- Reduce batch size: `config.api.batch_size: 50`
- Implement streaming predictions
- Use model quantization

---

## Rollback Plan

If deployment fails:

```bash
# Restore previous version
docker pull spornado-disease-prediction:0.9
docker run -p 5000:5000 spornado-disease-prediction:0.9

# Or manually restore
cp -r /backup/models/* models/
systemctl restart spornado-service
```

---

For more information, see [ARCHITECTURE.md](ARCHITECTURE.md) and [API_GUIDE.md](API_GUIDE.md).
