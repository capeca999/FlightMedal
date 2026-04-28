import cv2
import numpy as np


class ImageReader:
    async def read(self, file):
        contents = await file.read()
        array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("No se pudo leer la imagen")

        return image