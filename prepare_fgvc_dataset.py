from pathlib import Path
import shutil
import re


FGVC_DATA_DIR = Path("fgvc-aircraft-2013b/data")
OUTPUT_DIR = Path("datasets/aircraft_models")

SELECTED_FAMILIES = {
    "Boeing 737",
    "Boeing 777",
    "Boeing 787",
    "Airbus A320",
    "Airbus A330",
    "A320",
    "A380",
    "Boeing 747"
}


def clean_class_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r"[^a-zA-Z0-9]+", "_", name)
    return name.strip("_")


def parse_annotation_file(file_path: Path) -> list[tuple[str, str]]:
    items = []

    with file_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            image_id, label = line.split(" ", 1)
            items.append((image_id, label))

    return items


def copy_split(split: str):
    annotation_file = FGVC_DATA_DIR / f"images_family_{split}.txt"
    images_dir = FGVC_DATA_DIR / "images"

    items = parse_annotation_file(annotation_file)

    copied = 0

    for image_id, family in items:
        if family not in SELECTED_FAMILIES:
            continue

        class_name = clean_class_name(family)

        source_image = images_dir / f"{image_id}.jpg"
        target_dir = OUTPUT_DIR / split / class_name
        target_image = target_dir / f"{image_id}.jpg"

        target_dir.mkdir(parents=True, exist_ok=True)

        if source_image.exists():
            shutil.copy2(source_image, target_image)
            copied += 1

    print(f"{split}: {copied} imágenes copiadas")


def main():
    for split in ["train", "val", "test"]:
        copy_split(split)

    print("Dataset preparado correctamente.")


if __name__ == "__main__":
    main()