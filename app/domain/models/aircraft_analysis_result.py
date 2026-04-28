from dataclasses import dataclass
from typing import Optional

from app.domain.models.aircraft_detection import AircraftDetection


@dataclass
class AircraftAnalysisResult:
    media_type: str
    aircraft_detected: bool
    model: Optional[str]
    model_confidence: Optional[float]
    airline: Optional[str]
    airline_confidence: Optional[float]
    detections: list[AircraftDetection]