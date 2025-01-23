# Neustes Datenset UI-Control-C#-9 | 14.01.2025 | 2.375 Bilder

from roboflow import Roboflow
rf = Roboflow(api_key="i69F4gJn4h97e9d0s4XC")
project = rf.workspace("bildererkennung-im-bild").project("ui-control-c")
version = project.version(9)
dataset = version.download("yolov8")