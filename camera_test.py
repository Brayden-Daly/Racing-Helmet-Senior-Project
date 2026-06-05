#!/usr/bin/env python3
"""
MMlove USB Stereo Camera Test
- Shows LEFT and RIGHT camera feeds
- Works with most stereo USB webcams on Raspberry Pi / PC
- Press Q to quit

Install:
pip install opencv-python

Run:
python3 stereo_test.py
"""

import cv2

# -------------------------------------------------
# Try opening stereo camera
# -------------------------------------------------
# Most stereo USB cameras appear as ONE device
# with left/right images side-by-side.
#
# Usually camera index 0 works.
# If not, try 1, 2, etc.
# -------------------------------------------------

CAMERA_INDEX = 0

cap = cv2.VideoCapture(CAMERA_INDEX)

# Set resolution
# Many stereo cams output:
# 2560x720  (1280x720 per eye)
# 3840x1080 (1920x1080 per eye)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# FPS request
cap.set(cv2.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("Camera opened successfully.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    h, w, _ = frame.shape

    # Split stereo image into left/right
    mid = w // 2

    left_frame = frame[:, :mid]
    right_frame = frame[:, mid:]

    # Labels
    cv2.putText(left_frame, "LEFT CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(right_frame, "RIGHT CAMERA",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    # Show feeds
    cv2.imshow("Left Camera", left_frame)
    cv2.imshow("Right Camera", right_frame)

    # Combined preview
    cv2.imshow("Stereo Combined", frame)

    key = cv2.waitKey(1)

    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
