class VideoAnalysisService:
    def __init__(
        self,
        detector,
        model_classifier,
        airline_classifier,
        frame_extractor,
    ):
        self.detector = detector
        self.model_classifier = model_classifier
        self.airline_classifier = airline_classifier
        self.frame_extractor = frame_extractor

    async def analyze(self, file):
        frames = await self.frame_extractor.extract_key_frames(file, every_seconds=1)

        model_votes = []
        airline_votes = []
        detections_result = []

        for frame in frames:
            detections = self.detector.detect(frame)

            if not detections:
                continue

            best_detection = max(detections, key=lambda d: d.confidence)
            crop = self._crop_aircraft(frame, best_detection.box)

            model_votes.append(self.model_classifier.predict(crop))
            airline_votes.append(self.airline_classifier.predict(crop))
            detections_result.append(best_detection)

        if not model_votes:
            return {
                "media_type": "video",
                "aircraft_detected": False,
                "aircraft": None,
                "detections": []
            }

        final_model = self._majority_vote(model_votes)
        final_airline = self._majority_vote(airline_votes)

        return {
            "media_type": "video",
            "aircraft_detected": True,
            "aircraft": {
                "model": final_model.label,
                "model_confidence": final_model.confidence,
                "airline": final_airline.label,
                "airline_confidence": final_airline.confidence,
            },
            "detections": [d.to_dict() for d in detections_result]
        }

    def _crop_aircraft(self, frame, box):
        x1, y1, x2, y2 = box
        return frame[y1:y2, x1:x2]

    def _majority_vote(self, predictions):
        grouped = {}

        for prediction in predictions:
            grouped.setdefault(prediction.label, []).append(prediction.confidence)

        best_label = max(grouped, key=lambda label: sum(grouped[label]) / len(grouped[label]))

        return PredictionResult(
            label=best_label,
            confidence=sum(grouped[best_label]) / len(grouped[best_label])
        )