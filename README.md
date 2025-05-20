# README — `main.py`

## Descripción
`main.py` es un script **autocontenible** en Python que realiza **detección de objetos en tiempo real** con **YOLOv8** y muestra los resultados en una ventana de OpenCV (cajas, clases, confianza y FPS). Funciona con cualquier cámara disponible (webcam integrada, USB o Continuity Camera) o con archivos/streams de vídeo.

---

## Requisitos

| Paquete       | Versión sugerida | Instalación                  |
|---------------|-----------------|------------------------------|
| Python        | ≥ 3.9           | https://www.python.org/      |
| ultralytics   | ≥ 8.2           | `pip install ultralytics`    |
| opencv-python | ≥ 4.9           | `pip install opencv-python`  |

> **Nota:** `ultralytics` instala automáticamente PyTorch.  
> Si tu GPU es compatible con CUDA, PyTorch lo detectará y acelerará la inferencia.

---

## Pesos de YOLO

1. Descarga un modelo pre-entrenado (nano, ≈ 6 MB):
   ```bash
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
