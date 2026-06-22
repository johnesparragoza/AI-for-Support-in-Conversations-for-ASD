# Python 3.11 (not 3.13) so the 2024-era pinned stack has prebuilt wheels
# and doesn't fall back to slow/failing source builds.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install deps first so this layer is cached across app code changes
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# App code + Streamlit theme config + static assets
COPY app.py ./
COPY puzzle-buddy.png ./
COPY .streamlit/ ./.streamlit/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
