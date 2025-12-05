from fastapi.testclient import TestClient
from app.main import app

# TestClientの作成
client = TestClient(app)

# -------------------------------
# ルートエンドポイントのテスト
# -------------------------------
def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Titanic Survival Prediction API is running!"}

# -------------------------------
# 予測エンドポイントの正常系テスト
# -------------------------------
def test_predict_survival():
    test_data = {
        "Pclass": 3,
        "Sex": 1,
        "Age": 25,
        "C": 0,
        "Q": 0,
        "S": 1
    }
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert response.json()["prediction"] in ["Survived", "Not Survived"]

# -------------------------------
# 予測エンドポイントの異常系テスト
# -------------------------------
def test_predict_invalid_input():
    invalid_data = {
        "Pclass": 3,
        "Sex": 1,
        "Age": "invalid",  # 文字列 → バリデーションエラー
        "C": 0,
        "Q": 0,
        "S": 1
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422
