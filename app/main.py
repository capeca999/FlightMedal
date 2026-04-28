from fastapi import FastAPI, UploadFile, File, HTTPException

from app.aircraft_classifier import AircraftClassifier
from app.image_reader import ImageReader

app = FastAPI(title="Aircraft Model Recognition API")

classifier = AircraftClassifier("models/best.pt")
image_reader = ImageReader()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "Aircraft API funcionando"
    }


@app.post("/analyze")
async def analyze_aircraft(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Solo se permiten imágenes JPG, PNG o WEBP"
        )

    try:
        image = await image_reader.read(file)
        prediction = classifier.predict(image)

        return {
            "aircraft_detected": True,
            "prediction": prediction
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error)
        )