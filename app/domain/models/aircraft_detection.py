from dataclasses import dataclass


@dataclass
class AircraftDetection:
    box: list[int]
    confidence: float

    def to_dict(self):
        return {
            "box": self.box,
            "confidence": self.confidence
        }