# AI Resume Screening — FastAPI (default) + optional Streamlit
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dependency layer (cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime application files
COPY configs/ configs/
COPY src/ src/
COPY app/ app/

EXPOSE 8000 8501

# Default: FastAPI. For Streamlit, override the command (see README).
# Embedding model downloads on first screening use if not already cached.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
