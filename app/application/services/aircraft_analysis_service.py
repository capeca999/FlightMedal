class AircraftAnalysisService:
    def __init__(
        self,
        aircraft_detector,
        aircraft_model_classifier,
        airline_classifier,
        image_analysis_service,
        video_analysis_service,
    ):
        self.aircraft_detector = aircraft_detector
        self.aircraft_model_classifier = aircraft_model_classifier
        self.airline_classifier = airline_classifier
        self.image_analysis_service = image_analysis_service
        self.video_analysis_service = video_analysis_service

    async def analyze(self, file):
        if self._is_video(file.filename):
            return await self.video_analysis_service.analyze(file)

        return await self.image_analysis_service.analyze(file)

    def _is_video(self, filename: str) -> bool:
        return filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))