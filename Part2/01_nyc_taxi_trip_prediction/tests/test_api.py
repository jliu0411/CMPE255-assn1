from fastapi.testclient import TestClient
from app.main import app
client=TestClient(app)
def test_health(): assert client.get('/api/health').status_code==200
def test_invalid_coordinates():
    response=client.post('/api/predict',json={"pickup_latitude":0,"pickup_longitude":0,"dropoff_latitude":0,"dropoff_longitude":0,"pickup_datetime":"2016-01-01T12:00:00","passenger_count":1,"vendor_id":1});assert response.status_code==422
