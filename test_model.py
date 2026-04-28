from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"

# Cambia esta ruta por una imagen real de tu carpeta test
IMAGE_PATH = "datasets/aircraft_models/test/A320/0412223.jpg"


def main():
    image_path = Path(IMAGE_PATH)

    if not image_path.exists():
        raise FileNotFoundError(f"No existe la imagen: {image_path}")

    model = YOLO(MODEL_PATH)

    results = model.predict(
        source=str(image_path),
        verbose=False
    )

    result = results[0]

    class_id = int(result.probs.top1)
    confidence = float(result.probs.top1conf)
    label = result.names[class_id]

    print("Predicción:", label)
    print("Confianza:", round(confidence, 4))


if __name__ == "__main__":
    main()