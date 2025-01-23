# train.py: Füge den Code für das Training ein.

from ultralytics import YOLO

# Lade das "yolov8n.pt"
model = YOLO("C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\YOLO\\runs\detect\\train\\weights\\best.pt")
 

# Starte das Training | neue Parameter
model.train(data="C:\\Users\\jakob.derzapf\\source\\repos\\PythonProjekte\\YOLO\\UI-Control-C#-9\\config\\data.yaml", 
            epochs=500,               # Erhöhte Epochenanzahl für längeres Training
            batch=16,                # Mittelgroße Batch-Größe für GPU-Speicher und Stabilität
            #imgsz=640,               # Standardbildgröße bleibt 640x640
            lr0=0.005,               # Reduzierte Start-Lernrate für stabilere Konvergenz
            optimizer="AdamW",       # Spezifischer AdamW-Optimierer (gut für größere Datenmengen)
            warmup_epochs=1,         # Verlängerte Warm-up-Phase für besseren Start
            patience=100,             # Geduld erhöhen, um mehr Zeit für Verbesserungen zu lassen | wohl eher wenn es keine Verbesserungen mehr gibt!
            save=True,               # Modelle nach kompletten durchlauf speichern
           # save_period=2,           # Speicherintervall auf jede zweite Epoche setzen
            workers=8,               # Maximale Parallelisierung der Datenaufbereitung
            verbose=True,            # Detaillierte Trainingsinformationen anzeigen
            augment=True,            # Datenaugmentation aktivieren
            #device=0,
            #mosaic=1.0,              # Mosaic-Augmentierung für vielfältigere Trainingsdaten
            #mixup=0.2,               # MixUp-Datenaugmentation aktivieren
            #perspective=0.0,         # Perspektivische Verzerrung deaktivieren (UI-spezifisch)
            #hsv_h=0.01,              # Farbtonänderung minimieren (relevanter für UI-Bilder)
            #hsv_s=0.7,               # Sättigungsänderung für Varianz
            #hsv_v=0.4,               # Helligkeitsänderung für unterschiedliche Lichtbedingungen
            scale=0.5,               # Skaliere kleine Objekte stärker
            #degrees=15.0,            # Zufällige Drehung für unterschiedliche Ausrichtungen
            translate=0.1,           # Bildverschiebung zur Unterstützung teilweise sichtbarer Objekte
            #shear=5.0,               # Verzerrung für verschiedene Betrachtungswinkel
            #flipud=0.0,              # Keine vertikale Spiegelung, falls nicht relevant
            fliplr=0.5,              # Horizontale Spiegelung für mehr Variabilität
            bgr=0.0,                 # Keine Kanalumkehr für diese Anwendung
            copy_paste=0.0,          # Kein Copy-Paste für diesen Anwendungsfall
            copy_paste_mode="flip",  # Falls Copy-Paste verwendet wird, dann Flip-Modus
            auto_augment="randaugment",  # Automatische Erweiterungsstrategie für verbesserte Generalisierung
            erasing=0.4,             # Löschen von Bildteilen zur Förderung von Robustheit
            crop_fraction=1.0,       # Vollständiges Zuschneiden für fokussierte Objekterkennung 
            #classes=[0,1,2,3,6,9,10,11,13]   
            exist_ok = True        
            )




