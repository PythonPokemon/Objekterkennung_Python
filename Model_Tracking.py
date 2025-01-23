from ultralytics import YOLO
import cv2
import easyocr
from sort import Sort  # Sort-Bibliothek importieren (installiere sie mit `pip install sort-tracker`)

# 1. Initialisiere YOLO-Modell
model = YOLO("C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\runs\\detect\\train\\weights\\best.pt")  # Pfad zu deinem trainierten YOLO-Modell

# 2. Initialisiere Sort-Tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)  # Tracking-Parameter (anpassbar)

# 3. Bild laden und YOLO-Inferenz durchführen
image_path = "c:\\Users\\jakob.derzapf\\Downloads\\UI.PNG"  # Pfad zum Testbild
image = cv2.imread(image_path)

if image is None:
    print(f"Fehler: Bild konnte nicht geladen werden: {image_path}")
    exit()

results = model.predict(source=image, save=True, save_txt=True)  # Inferenz durchführen
print("YOLO-Ergebnisse geladen.")

# Klassennamen (anpassen, falls nötig)
class_names = ["Button", "Checkbox", "Dropdown-Menu", "Input", "Item", "Label",
               "Radiobutton", "Spinner-UpDown", "Textblock"]

# 4. Ergebnisse von YOLO verarbeiten
reader = easyocr.Reader(['en'])  # EasyOCR-Reader für Englisch initialisieren

# Liste für Tracking-Eingabe vorbereiten
detections = []  # Format: [[x1, y1, x2, y2, score], ...]

for result in results[0].boxes:  # Schleife durch die erkannten Objekte
    # Extrahiere die Bounding Box und andere Informationen
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Begrenzungsrahmen
    cls = int(result.cls[0])  # Klasse des erkannten Objekts
    confidence = result.conf[0]  # Konfidenz (Confidence Score)

    # Detektierte Klasse und Name
    detected_class = class_names[cls]
    print(f"Erkannt: {detected_class}, Confidence: {confidence:.2f}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Bounding Box und Klassennamen ins Bild zeichnen
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(image, f"{detected_class} ({confidence:.2f})", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Füge die Bounding Box und Confidence zum Tracking hinzu
    detections.append([x1, y1, x2, y2, confidence])

# 5. Sort-Tracking anwenden
tracked_objects = tracker.update(detections)  # Update-Tracker mit aktuellen Erkennungen

# 6. Gehe durch die verfolgten Objekte
for obj in tracked_objects:
    x1, y1, x2, y2, obj_id = map(int, obj)  # Extrahiere Tracking-Informationen
    print(f"Tracking-ID: {obj_id}, Box: ({x1}, {y1}, {x2}, {y2})")

    # Zeichne Tracking-ID ins Bild
    cv2.putText(image, f"ID: {obj_id}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

# 7. OCR auf relevanten Klassen (z. B. Label, Textbox)
for result in results[0].boxes:  # Schleife durch die erkannten Objekte
    x1, y1, x2, y2 = map(int, result.xyxy[0])  # Begrenzungsrahmen
    cls = int(result.cls[0])  # Klasse des erkannten Objekts
    detected_class = class_names[cls]

    if detected_class in ["Button"]:
        # Ausschneiden des Bereichs
        cropped_region = image[y1:y2, x1:x2]

        # OCR-Ergebnisse abrufen
        ocr_results = reader.readtext(cropped_region)
        for (ocr_bbox, text, ocr_confidence) in ocr_results:
            print(f"OCR erkannt: {text}, OCR-Confidence: {ocr_confidence:.2f}")
            # Zeichne erkannte OCR-Texte ins Bild
            cv2.putText(image, text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 8. Ergebnis speichern und anzeigen
output_path = "c:\\Users\\jakob.derzapf\\Downloads\\output_tracking.png"
cv2.imwrite(output_path, image)
print(f"Ergebnisbild gespeichert unter: {output_path}")

cv2.imshow("Ergebnisse", image)  # Bild anzeigen
cv2.waitKey(0)
cv2.destroyAllWindows()
