import easyocr

# Absoluter Pfad zum Testbild
image_path = "c:\\Users\\jakob.derzapf\\Downloads\\UI.PNG"

reader = easyocr.Reader(['en'])  # Initialisiere den EasyOCR-Reader für Englisch
results = reader.readtext(image_path)  # Führe Texterkennung auf dem angegebenen Bild durch

# Ausgabe der Ergebnisse
for (bbox, text, confidence) in results:
    print(f"Text: {text}, Confidence: {confidence}")
