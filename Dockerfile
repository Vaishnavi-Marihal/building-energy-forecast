FROM node:22-slim AS frontend
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY sql/ sql/
# deploy/ holds the demo readings, the trained models and the data notice.
# The API reads the models from artifacts/; the loader reads the readings.
COPY deploy/ deploy/
RUN ln -s /app/deploy/model /app/artifacts
COPY --from=frontend /app/frontend/dist frontend/dist
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT:-8000}"]
