from fastapi import APIRouter, UploadFile, File, HTTPException

from app.infrastructure.ai.aircraft_detector import AircraftDetector
from app.infrastructure.ai.aircraft_model_classifier import AircraftModelClassifier
from app.infrastructure.ai.airline_classifier import AirlineClassifier
from app.infrastructure.media.image_reader import ImageReader
from app.infrastructure.media.video_frame_extractor import VideoFrameExtractor
from app.application.services.image_analysis_service import ImageAnalysisService
from app.application.services.video_analysis_service import VideoAnalysisService
from app.application.services.aircraft_analysis_service import AircraftAnalysisService

router = APIRouter()

detector = AircraftDetector("models/aircraft_detector.pt")
model_classifier = AircraftModelClassifier("models/aircraft_model_classifier.pt")
airline_classifier = AirlineClassifier("models/airline_classifier.pt")

image_service = ImageAnalysisService(
    detector=detector,
    model_classifier=model_classifier,
    airline_classifier=airline_classifier,
    image_reader=ImageReader()
)

video_service = VideoAnalysisService(
    detector=detector,
    model_classifier=model_classifier,
    airline_classifier=airline_classifier,
    frame_extractor=VideoFrameExtractor()
)

analysis_service = AircraftAnalysisService(
    aircraft_detector=detector,
    aircraft_model_classifier=model_classifier,
    airline_classifier=airline_classifier,
    image_analysis_service=image_service,
    video_analysis_service=video_service
)


@router.post("/analyze")
async def analyze_aircraft(file: UploadFile = File(...)):
    if not file.content_type:
        raise HTTPException(status_code=400, detail="Invalid file")

    allowed_types = [
        "image/jpeg",
        "image/png",
        "image/webp",
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo"
    ]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}"
        )

    return await analysis_service.analyze(file)