import time

import cv2
from ultralytics import YOLO


def main():
    model = YOLO("yolo11n.pt")
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check Windows camera permissions and camera index."
        )

    previous_time = time.perf_counter()

    while True:
        success, frame = camera.read()
        if not success:
            print("Could not read a webcam frame.")
            break

        results = model.predict(source=frame, conf=0.40, verbose=False)
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

        cv2.imshow("YOLO11 + OpenCV - Webcam Detection", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
