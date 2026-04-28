from ultralytics import YOLO
from app.domain.models.aircraft_detection import AircraftDetection


class AircraftDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image) -> list[AircraftDetection]:
        results = self.model.predict(image, conf=0.35, verbose=False)

        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])

                detections.append(
                    AircraftDetection(
                        box=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=confidence
                    )
                )

        return detections