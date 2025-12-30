from ultralytics import YOLO
import requests
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
try:
    results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)
finally:
    url = 'https://xn-b.suanjiayun.com:4333/container/api/projects/67fd13f6e2946e9a0ea9615a/instances/694e33a369d977df35c92a4c/67fd13f6e2946e9a0ea9615c/shutDown'
    headers = {
        'Authorization': '2FSVX1azeZrQJmqTG7mTItrJ0u3kAQvYEhIqxsohHOTDNglE1BF7N1JnR0tMZZ6L',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers)

    print("状态码：", response.status_code)
    print("响应内容：", response.text)
