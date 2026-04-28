from ultralytics import YOLO


class AircraftClassifier:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image):
        results = self.model.predict(image, verbose=False)
        result = results[0]

        class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = result.names[class_id]

        return {
            "model": label,
            "confidence": round(confidence, 4)
        }