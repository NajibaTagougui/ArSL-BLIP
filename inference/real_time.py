"""
Real-Time Webcam Inference for ArSL-BLIP

Authors: Najiba Tagougui, Ansar Hani, Monji Kherallah
Year: 2026

Usage:
    python inference/real_time.py
    python inference/real_time.py --checkpoint models/arsl-blip --camera 0
"""

import argparse
import time
import cv2
import numpy as np
from PIL import Image

from models.blip_arabic import ArabicSignRecognizer


def run_webcam(model_path: str, camera_id: int = 0, threshold: float = 0.85) -> None:
    """Run real-time inference from webcam feed."""
    recognizer = ArabicSignRecognizer(model_path=model_path, confidence_threshold=threshold)

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    print("Press 'q' to quit | 'space' to capture and predict")

    last_result = None
    frame_count = 0
    fps_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        elapsed = time.time() - fps_start
        fps = frame_count / elapsed if elapsed > 0 else 0

        # Run prediction every 10 frames for speed
        if frame_count % 10 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            last_result = recognizer.predict(pil_img)

        # Overlay results
        display = frame.copy()
        if last_result:
            letter = last_result.get("letter", "?") or "?"
            arabic = last_result.get("arabic_char", "") or ""
            conf = last_result.get("confidence", 0.0)
            accepted = last_result.get("accepted", False)

            color = (0, 200, 0) if accepted else (0, 100, 255)
            cv2.putText(display, f"Letter: {letter}", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(display, f"Arabic: {arabic}", (20, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
            cv2.putText(display, f"Conf: {conf:.0%}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            status = "ACCEPTED" if accepted else "LOW CONFIDENCE"
            cv2.putText(display, status, (20, 185),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(display, f"FPS: {fps:.1f}", (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)

        cv2.imshow("ArSL-BLIP Real-Time", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Stream closed.")


def main():
    parser = argparse.ArgumentParser(description="Real-time ArSL-BLIP inference")
    parser.add_argument("--checkpoint", type=str, default="models/arsl-blip")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()
    run_webcam(args.checkpoint, args.camera, args.threshold)


if __name__ == "__main__":
    main()
