FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

ENV PYTHONPATH=/app/backend
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "streamlit run backend/app/ui.py --server.address=0.0.0.0 --server.port=${PORT} --server.headless=true"]
