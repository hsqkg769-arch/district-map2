# デプロイ手順

このStreamlitアプリケーションをデプロイする方法を説明します。

## Streamlit Cloudへのデプロイ（推奨）

Streamlit Cloudは無料で利用でき、GitHubリポジトリから直接デプロイできます。

### 前提条件

1. GitHubアカウント
2. Streamlit Cloudアカウント（https://streamlit.io/cloud で無料登録可能）

### 手順

1. **GitHubリポジトリにプッシュ**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <あなたのGitHubリポジトリURL>
   git push -u origin main
   ```

2. **Streamlit Cloudでデプロイ**
   - https://share.streamlit.io/ にアクセス
   - "New app" をクリック
   - GitHubアカウントでログイン
   - リポジトリ、ブランチ、メインファイル（`app.py`）を選択
   - "Deploy!" をクリック

3. **環境変数の設定（必要に応じて）**
   - Streamlit Cloudのダッシュボードで "Settings" → "Secrets" から環境変数を設定可能

### 注意事項

- `data/` ディレクトリ内のファイルはリポジトリに含まれている必要があります
- 大きなファイル（>100MB）はGit LFSを使用するか、外部ストレージを検討してください
- 初回デプロイには数分かかる場合があります

## その他のデプロイ方法

### Dockerを使用したデプロイ

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# システム依存関係のインストール
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

# Streamlitの起動
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### ローカルでの実行

```bash
# 依存関係のインストール
pip install -r requirements.txt

# アプリケーションの起動
streamlit run app.py
```

## トラブルシューティング

### GeoPandas関連のエラー

GeoPandasはGDALなどのシステムライブラリに依存しています。Streamlit Cloudでは自動的にインストールされますが、ローカル環境では手動でインストールが必要な場合があります。

### メモリ不足エラー

大きなShapefileファイルを読み込む場合、メモリ不足になる可能性があります。Streamlit Cloudの無料プランでは制限があるため、必要に応じてデータの最適化を検討してください。

