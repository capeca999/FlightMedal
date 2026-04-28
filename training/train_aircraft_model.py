from ultralytics import YOLO


def main():
    model = YOLO("yolo11n-cls.pt")

    model.train(
        data="datasets/aircraft_models",
        epochs=30,
        imgsz=224,
        batch=16,
        name="aircraft_model_classifier"
    )


if __name__ == "__main__":
    main()