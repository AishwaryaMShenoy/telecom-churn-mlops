FROM python:3.12-slim

WORKDIR /app

COPY requirements-docker.txt .

RUN pip install --no-cache-dir -r requirements-docker.txt

COPY api ./api
COPY src ./src
COPY models ./models
COPY data ./data
COPY config ./config

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]