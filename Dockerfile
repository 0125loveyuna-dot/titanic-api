# Pythonの公式イメージ（3.9-slim）を使用
FROM python:3.9-slim

# 作業ディレクトリを /app に設定
WORKDIR /app

# モデルファイルとFastAPIアプリをコンテナにコピー
COPY titanic_model.joblib /app/titanic_model.joblib
COPY app/ /app/app/

# 必要なPythonライブラリをインストール
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# FastAPI が待ち受けるポート
EXPOSE 8000

# コンテナ起動時にFastAPIアプリを起動
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
