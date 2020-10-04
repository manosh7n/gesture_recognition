import cv2
import numpy as np
from tracker.hand_tracker import HandTracker


WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "models/palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "models/hand_landmark.tflite"
ANCHORS_PATH = "models/anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2

cv2.namedWindow(WINDOW)
capture = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*"MJPG")
writer = cv2.VideoWriter('videos/output_01.avi', fourcc, 24, (640, 480), True)

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.1,
    box_enlarge=1.1
)

while True:
    hasFrame, frame = capture.read()

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    points, bbox = detector(image)

    if bbox is not None:
        bbox_pt = np.array(bbox, dtype=np.int)
        bbox_pt = np.reshape(bbox_pt, (-1, 4, 2))
        rect = cv2.boundingRect(bbox_pt)
        x, y, w, h = np.abs(rect)
        crop = frame[y:y + h, x:x + w].copy()
        cv2.polylines(frame, bbox_pt, 1, POINT_COLOR, THICKNESS)
        cv2.imshow('Crop', crop)

    if writer is not None:
        writer.write(frame)

    cv2.imshow(WINDOW, frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

if writer is not None:
    writer.release()

capture.release()
cv2.destroyAllWindows()
