# hungtrab:stock_prediction:latest/Dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/app

COPY ./app /app/app
COPY ./data /app/data
COPY ./database /app/database
COPY ./model_training /app/model_training
COPY ./models_store /app/models_store
COPY ./ui /app/ui
COPY requirements.txt /app/requirements.txt
COPY ./models_research /app/models_research

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
