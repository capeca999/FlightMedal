class ImageAnalysisService:
    def __init__(
        self,
        detector,
        model_classifier,
        airline_classifier,
        image_reader,
    ):
        self.detector = detector
        self.model_classifier = model_classifier
        self.airline_classifier = airline_classifier
        self.image_reader = image_reader

    async def analyze(self, file):
        image = await self.image_reader.read(file)

        detections = self.detector.detect(image)

        if not detections:
            return {
                "media_type": "image",
                "aircraft_detected": False,
                "aircraft": None,
                "detections": []
            }

        best_detection = max(detections, key=lambda d: d.confidence)
        aircraft_crop = self._crop_aircraft(image, best_detection.box)

        model_prediction = self.model_classifier.predict(aircraft_crop)
        airline_prediction = self.airline_classifier.predict(aircraft_crop)

        return {
            "media_type": "image",
            "aircraft_detected": True,
            "aircraft": {
                "model": model_prediction.label,
                "model_confidence": model_prediction.confidence,
                "airline": airline_prediction.label,
                "airline_confidence": airline_prediction.confidence,
            },
            "detections": [d.to_dict() for d in detections]
        }

    def _crop_aircraft(self, image, box):
        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2]