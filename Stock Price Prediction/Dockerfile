# index_stock_predictor_mlops/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Tạo các thư mục cần thiết bên trong image TRƯỚC KHI COPY hoặc MOUNT
RUN mkdir -p /app/app

COPY ./app /app/app
COPY ./data /app/data
COPY ./database /app/database
COPY ./model_training /app/model_training
COPY ./models_store /app/models_store
COPY ./ui /app/ui
COPY requirements.txt /app/requirements.txt

EXPOSE 8000

# CMD mặc định cho API server
CMD ["python", "-m", "uvicorn", "app.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
