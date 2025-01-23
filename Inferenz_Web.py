from flask import Flask, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)
model = YOLO("C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\runs\\detect\\train\\weights\\best.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    img_path = f"./uploads/{file.filename}"
    file.save(img_path)

    # Inferenz ausführen
    results = model.predict(source=img_path)

    # Ergebnisse formatieren
    predictions = []
    for box in results[0].boxes:
        predictions.append({
            "class": model.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "coordinates": box.xyxy[0].tolist()
        })

    return jsonify({"predictions": predictions})

if __name__ == '__main__':
    app.run(debug=True)
