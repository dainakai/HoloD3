"""Run direct production YOLO inference from Python."""

from pathlib import Path

import cv2

from holod3 import ParticleDetector

image = Path("data/demo/experimental/minip/005302.png")
detector = ParticleDetector(device="0")
detections = detector.predict(image)
for detection in detections:
    print(detection.to_dict())

output = Path("runs/examples/detect_minip.png")
output.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(output), detector.annotate(image, detections))
