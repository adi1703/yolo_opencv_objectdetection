import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11 + OpenCV image detection")
    parser.add_argument("--source", required=True, help="Path to an input image")
    parser.add_argument("--output", default="outputs/detected_image.jpg", help="Output image path")
    parser.add_argument("--conf", type=float, default=0.40, help="Confidence threshold")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Image not found: {source}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolo11n.pt")
    image = cv2.imread(str(source))

    if image is None:
        raise ValueError("OpenCV could not read the image.")

    results = model.predict(source=image, conf=args.conf, verbose=False)
    annotated = results[0].plot()

    cv2.imwrite(str(output), annotated)
    print(f"Saved result to: {output.resolve()}")


if __name__ == "__main__":
    main()
