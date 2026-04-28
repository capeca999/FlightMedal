import cv2
import tempfile


class VideoFrameExtractor:
    async def extract_key_frames(self, file, every_seconds: int = 1):
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
            temp_file.write(contents)
            temp_path = temp_file.name

        video = cv2.VideoCapture(temp_path)

        fps = video.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * every_seconds)

        frames = []
        frame_index = 0

        while True:
            success, frame = video.read()

            if not success:
                break

            if frame_index % frame_interval == 0:
                frames.append(frame)

            frame_index += 1

        video.release()

        return frames