from ultralytics import YOLO
import cv2

# Lade das Modell
model = YOLO("C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\runs\\detect\\train\\weights\\best.pt")

# Webcam starten
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Vorhersage
    results = model.predict(source=frame, show=True)

    # Beende mit der Taste 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
