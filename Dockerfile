# Lightweight container for the v2 Streamlit demo
FROM python:3.11-slim

# Avoid creating .pyc files and force stdout/stderr flush
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY . /app

# Install dependencies: prefer requirements.txt if present
RUN pip install --upgrade pip setuptools wheel
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Expose Streamlit port
EXPOSE 8501

# Default command: run the Streamlit demo
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.headless=true"]
