# 必要なライブラリをインポートします。
# FastAPI: Web APIを構築するためのフレームワーク
# BaseModel: APIが受け取るデータの形を定義するためのPydanticのクラス
# joblib: モデルファイルの読み込み用
# pandas: モデルが期待する入力形式（DataFrame）に変換するため
# 必要なライブラリのインポート
from fastapi import FastAPI                 # Web APIを構築するためのフレームワーク
from pydantic import BaseModel              # APIが受け取るデータの形を定義するため
import joblib                                # モデルファイルの読み込み用
import pandas as pd                          # 入力データをDataFrameに変換
import os

# MLflowのインポート（モデル情報のトラッキング用）
import mlflow
import mlflow.sklearn

# FastAPIアプリケーションのインスタンス作成
app = FastAPI()

# ----------------- モデルのロード -----------------
# main.pyはappディレクトリ内にあるため、2つ上のディレクトリがプロジェクトルート
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(project_root, 'titanic_model.joblib')

try:
    model = joblib.load(model_path)
    print(f"AIモデルを {model_path} から正常にロードしました。")
except FileNotFoundError:
    print(f"エラー: AIモデルファイルが見つかりません。パス: {model_path}")
    print("create_model.py を実行してモデルを作成してください。")
    # 起動時にエラーを出してユーザーに通知
    raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")
# ----------------------------------------------------

# ----------------- MLflowの設定 -----------------
mlflow.set_experiment("Titanic Survival Prediction API")  # 実験名を設定

with mlflow.start_run():  # ひとつの実験を開始
    mlflow.log_param("model_name", "RandomForestClassifier")
    mlflow.log_param("model_version", "1.0")               # 適当なバージョン情報
    mlflow.log_param("loaded_from", model_path)
    # 実際にはモデル学習時にMLflowを使うのが理想
    # 今回はAPIがロードするモデル情報を記録する例として

# ----------------- 入力データ形式の定義 -----------------
class InputData(BaseModel):
    Pclass: int  # 客室クラス (1, 2, 3)
    Sex: int     # 性別 (0:男性, 1:女性)
    Age: int     # 年齢
    C: int       # 乗船港 Cherbourg
    Q: int       # 乗船港 Queenstown
    S: int       # 乗船港 Southampton

# モデルが予測に使う特徴量の順序
features = ["Pclass", "Sex", "Age", "C", "Q", "S"]

# ----------------- 予測APIエンドポイント -----------------
@app.post("/predict")
def predict_survival(passenger: InputData):
    # モデルが受け取る形式（DataFrame）へ変換
    input_data = pd.DataFrame([passenger.dict()], columns=features)

    # 予測
    prediction = model.predict(input_data)[0]

    # 結果を人間向けテキストに変換
    survival_status = "Survived" if prediction == 1 else "Not Survived"

    return {"prediction": survival_status}

# ----------------- ヘルスチェックエンドポイント -----------------
@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API is running!"}