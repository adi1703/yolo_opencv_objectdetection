import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="YOLO11 + OpenCV video detection")
    parser.add_argument("--source", required=True, help="Input video path")
    parser.add_argument("--output", default="outputs/detected_video.mp4")
    parser.add_argument("--conf", type=float, default=0.40)
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"Video not found: {source}")

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    if input_fps <= 0:
        input_fps = 30.0

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        input_fps,
        (width, height),
    )

    model = YOLO("yolo11n.pt")
    previous_time = time.perf_counter()

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(source=frame, conf=args.conf, verbose=False)
        annotated = results[0].plot()

        current_time = time.perf_counter()
        fps = 1.0 / max(current_time - previous_time, 1e-6)
        previous_time = current_time

        cv2.putText(
            annotated,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        writer.write(annotated)

    cap.release()
    writer.release()
    print(f"Saved result to: {output.resolve()}")


if __name__ == "__main__":
    main()
