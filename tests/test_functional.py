
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from waterflow.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_page_loads(client):
    res = client.get("/")
    assert res.status_code == 200
    assert b"prediction form" or b"Formulaire" in res.data

def test_prediction_returns_result(client):
    data = {
        "ph": "7.0",
        "Hardness": "150",
        "Solids": "8000",
        "Chloramines": "7",
        "Sulfate": "250",
        "Conductivity": "420",
        "Organic_carbon": "6",
        "Trihalomethanes": "80",
        "Turbidity": "3.0"
    }
    res = client.post("/", data=data)
    assert res.status_code == 200

    html = res.data.decode("utf-8")
    assert "Résultat" in html or "Result" in html

