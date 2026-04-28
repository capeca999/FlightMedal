import cv2
import numpy as np


class ImageReader:
    async def read(self, file):
        contents = await file.read()
        np_array = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not read image")

        return image