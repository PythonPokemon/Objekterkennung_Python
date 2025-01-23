from ultralytics import YOLO
import cv2

# Lade das Modell
model = YOLO("C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\runs\\detect\\train\\weights\\best.pt")

# Pfad zum Video
video_path = "path/to/your/video.mp4"
output_path = "path/to/output/video.mp4"

# Öffne das Video
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Schreibe das Video
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Frameweise Inferenz
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Vorhersage für den Frame
    results = model.predict(source=frame)

    # Zeichne die Ergebnisse
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinaten des Begrenzungsrahmens
        conf = box.conf[0]                     # Konfidenz
        cls = int(box.cls[0])                  # Klassenindex
        label = f"{model.names[cls]} {conf:.2f}"  # Klassenname und Konfidenz

        # Zeichne Begrenzungsrahmen und Label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Speichere den Frame im Output
    out.write(frame)

# Schließe Ressourcen
cap.release()
out.release()
cv2.destroyAllWindows()
