from ultralytics import YOLO
from app.domain.models.aircraft_prediction import PredictionResult


class AirlineClassifier:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def predict(self, image) -> PredictionResult:
        results = self.model.predict(image, verbose=False)
        result = results[0]

        top_class_id = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        label = result.names[top_class_id]

        return PredictionResult(
            label=label,
            confidence=confidence
        )