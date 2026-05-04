#!/usr/bin/env python3
import time
import threading

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

CANNY_LOW = 20
CANNY_HIGH = 70
EDGE_THICKNESS = 2
JPEG_QUALITY = 70

app = Flask(__name__)

# Create both cameras explicitly
cam0 = Picamera2(camera_num=0)
cam1 = Picamera2(camera_num=1)

config0 = cam0.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    controls={"FrameRate": 30}
)

config1 = cam1.create_video_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
    controls={"FrameRate": 30}
)

cam0.configure(config0)
cam1.configure(config1)

cam0.start()
cam1.start()
time.sleep(2)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

encode_params = [
    int(cv2.IMWRITE_JPEG_QUALITY),
    JPEG_QUALITY
]


def process_frame(frame_bgr):
    display = frame_bgr.copy()

    small = cv2.resize(
        frame_bgr,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA
    )

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = clahe.apply(gray)

    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharp = cv2.addWeighted(gray, 1.7, blurred, -0.7, 0)

    edges = cv2.Canny(sharp, CANNY_LOW, CANNY_HIGH)

    if EDGE_THICKNESS > 1:
        kernel = np.ones((EDGE_THICKNESS, EDGE_THICKNESS), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    edges_full = cv2.resize(
        edges,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_NEAREST
    )

    display[edges_full > 0] = (0, 0, 255)

    return display


def generate_frames(camera, camera_name):
    t0 = time.time()
    frames = 0

    while True:
        frame = camera.capture_array()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        output = process_frame(frame)

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            print(f"{camera_name} FPS: {frames / (now - t0):.2f}")
            t0 = now
            frames = 0

        ok, buffer = cv2.imencode(".jpg", output, encode_params)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buffer.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Dual Camera Edge Overlay</title>
      </head>
      <body style="background:#111;color:#eee;font-family:sans-serif;text-align:center;">
        <h2>Dual Camera Red Edge Overlay</h2>

        <div style="display:flex;justify-content:center;gap:20px;flex-wrap:wrap;">
          <div>
            <h3>Camera 0</h3>
            <img src="/video_feed0" width="640">
          </div>

          <div>
            <h3>Camera 1</h3>
            <img src="/video_feed1" width="640">
          </div>
        </div>
      </body>
    </html>
    """


@app.route("/video_feed0")
def video_feed0():
    return Response(
        generate_frames(cam0, "Camera 0"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/video_feed1")
def video_feed1():
    return Response(
        generate_frames(cam1, "Camera 1"),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        cam0.stop()
        cam1.stop()
