FROM python:3.11-slim

WORKDIR /app

# システム依存関係のインストール（GeoPandas用）
RUN apt-get update && apt-get install -y \
    libgdal-dev \
    gdal-bin \
    proj-data \
    proj-bin \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Python依存関係のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションファイルのコピー
COPY . .

# ポート8501を公開
EXPOSE 8501

# Streamlitの起動
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

