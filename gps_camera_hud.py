#!/usr/bin/env python3
import math
import time
import threading
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import serial
import pynmea2
from picamera2 import Picamera2

# ============================================================
# GLOBAL CONFIG
# ============================================================

# ----------------------------
# GPS
# ----------------------------
PORT = "/dev/ttyACM0"
BAUD = 9600
R_EARTH = 6371000.0

GPS_MIN_UPDATE_DIST_M = 2.5
GPS_LOOP_CLOSE_DIST_M = 10.0
GPS_MIN_LOOP_POINTS = 35
GPS_MIN_LOOP_LENGTH_M = 120.0
GPS_TURN_ANGLE_DEG = 22.0
GPS_MIN_TURN_SPACING_M = 12.0
GPS_MAX_MATCH_DIST_M = 20.0
GPS_LOOP_PROGRESS_SLACK_M = 6.0

# ----------------------------
# Live Camera
# ----------------------------
CAMERA_NUM = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# No output file
WRITE_OUTPUT = False
OUTPUT_PATH = ""

SHOW_ORIGINAL_VIDEO = True
SHOW_EDGE_VIDEO = True
SHOW_HUD = True
SHOW_GPS_MAP = True

FRAME_BUFFER_SIZE = 8

# ----------------------------
# ROI
# ----------------------------
ROI = dict(
    top_y=0.08,
    bottom_y=0.90,
    left_bottom=0.0,
    right_bottom=1.0,
    left_top=0.10,
    right_top=0.90,
)

CAR_MASK = dict(
    enabled=True,
    left=0.14,
    right=0.86,
    top_y=0.58,
)

BOTTOM_CUT_FRAC = 0.88

# ----------------------------
# Paint detection
# ----------------------------
USE_YELLOW = False

WHITE_V_MIN = 200
WHITE_S_MAX = 30

YELLOW_LOW = (12, 90, 120)
YELLOW_HIGH = (45, 255, 255)

# ----------------------------
# Preprocessing
# ----------------------------
BLUR_K = 5

# ----------------------------
# Thin-edge extraction
# ----------------------------
GRAD_THRESH = 34

CENTER_BLOCK_L = 0.43
CENTER_BLOCK_R = 0.57

# ============================================================
# HUD CONFIG
# ============================================================

DISPLAY_W = 1280
DISPLAY_H = 960
START_DIST_M = 100.0

COL_BG = (0, 0, 0)
COL_GREEN = (0, 200, 0)
COL_YELLOW = (0, 220, 255)
COL_RED = (0, 0, 220)
COL_WHITE = (220, 220, 220)

# ============================================================
# SHARED STATE
# ============================================================

hud_lock = threading.Lock()
gps_lock = threading.Lock()
frame_lock = threading.Lock()

turn_dir = None
distance_m = 0.0
active = False

gps_running = True
video_running = True

gps_status = {
    "lat": None,
    "lon": None,
    "sats": None,
    "hdop": None,
    "loop_closed": False,
    "track_len_m": 0.0,
    "loop_len_m": 0.0,
    "status_text": "Waiting for GPS fix...",
}

points_xy = []
video_fps = 30.0
video_width = None
video_height = None
frame_queue = deque(maxlen=FRAME_BUFFER_SIZE)

# ============================================================
# HUD HELPERS
# ============================================================

def alert_color(dist_m):
    t = 1.0 - max(0.0, min(1.0, dist_m / START_DIST_M))
    b = int(COL_YELLOW[0] + (COL_RED[0] - COL_YELLOW[0]) * t)
    g = int(COL_YELLOW[1] + (COL_RED[1] - COL_YELLOW[1]) * t)
    r = int(COL_YELLOW[2] + (COL_RED[2] - COL_YELLOW[2]) * t)
    return (b, g, r)


def trigger_turn(direction, distance):
    global turn_dir, distance_m, active
    with hud_lock:
        turn_dir = direction
        distance_m = max(0.0, float(distance))
        active = True


def clear_turn():
    global turn_dir, distance_m, active
    with hud_lock:
        turn_dir = None
        distance_m = 0.0
        active = False


def draw_arrow(img, cx, cy, size, direction, color):
    if direction == "LEFT":
        pts = np.array([
            [cx + size, cy - size],
            [cx, cy],
            [cx + size, cy + size]
        ], dtype=np.int32)
    else:
        pts = np.array([
            [cx - size, cy - size],
            [cx, cy],
            [cx - size, cy + size]
        ], dtype=np.int32)
    cv2.polylines(img, [pts], False, color, 4, cv2.LINE_AA)


def render_hud(frame):
    frame[:] = COL_BG
    W, H = DISPLAY_W, DISPLAY_H

    with hud_lock:
        local_turn = turn_dir
        local_dist = distance_m
        local_active = active
        local_status = gps_status.copy()

    if local_active and local_turn:
        col = alert_color(local_dist)

        arrow_cx = W // 4 if local_turn == "LEFT" else 3 * W // 4
        draw_arrow(frame, arrow_cx, H // 2 - 30, 36, local_turn, col)

        label = f"TURN {local_turn}"
        (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 2)
        cv2.putText(frame, label,
                    ((W - lw) // 2, H // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, col, 2, cv2.LINE_AA)

        dist_text = f"{max(local_dist, 0):.0f} m"
        (dw, _), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 2.4, 3)
        cv2.putText(frame, dist_text,
                    ((W - dw) // 2, H // 2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.4, col, 3, cv2.LINE_AA)
    else:
        (tw, _), _ = cv2.getTextSize("TRACK CLEAR", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 1)
        cv2.putText(frame, "TRACK CLEAR",
                    ((W - tw) // 2, H // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COL_GREEN, 1, cv2.LINE_AA)

    info_lines = [
        f"GPS: {local_status['status_text']}",
        f"Lat: {local_status['lat'] if local_status['lat'] is not None else '--'}   Lon: {local_status['lon'] if local_status['lon'] is not None else '--'}",
        f"Sats: {local_status['sats'] if local_status['sats'] is not None else '--'}   HDOP: {local_status['hdop'] if local_status['hdop'] is not None else '--'}",
        f"Track: {local_status['track_len_m']:.1f} m   Loop closed: {'YES' if local_status['loop_closed'] else 'NO'}   Loop: {local_status['loop_len_m']:.1f} m",
    ]

    y0 = H - 170
    for i, txt in enumerate(info_lines):
        cv2.putText(frame, txt, (40, y0 + i * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COL_WHITE, 2, cv2.LINE_AA)

# ============================================================
# GPS MAP
# ============================================================

def render_gps_map():
    map_w = 800
    map_h = 800
    img = np.zeros((map_h, map_w, 3), dtype=np.uint8)
    img[:] = (20, 20, 20)

    with gps_lock:
        pts = list(points_xy)
        local_status = gps_status.copy()

    if not pts:
        cv2.putText(img, "Waiting for GPS points...", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 220), 2, cv2.LINE_AA)
        return img

    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)

    min_x, max_x = float(xs.min()), float(xs.max())
    min_y, max_y = float(ys.min()), float(ys.max())

    span_x = max(max_x - min_x, 10.0)
    span_y = max(max_y - min_y, 10.0)
    pad = 40.0

    scale_x = (map_w - 2 * pad) / span_x
    scale_y = (map_h - 2 * pad) / span_y
    scale = min(scale_x, scale_y)

    draw_pts = []
    for x, y in pts:
        px = int((x - min_x) * scale + pad)
        py = int(map_h - ((y - min_y) * scale + pad))
        draw_pts.append((px, py))

    if len(draw_pts) >= 2:
        cv2.polylines(img, [np.array(draw_pts, dtype=np.int32)], False, (0, 255, 255), 2, cv2.LINE_AA)

    cv2.circle(img, draw_pts[0], 6, (0, 255, 0), -1)
    cv2.putText(img, "START", (draw_pts[0][0] + 8, draw_pts[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.circle(img, draw_pts[-1], 7, (0, 0, 255), -1)
    cv2.putText(img, "NOW", (draw_pts[-1][0] + 8, draw_pts[-1][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(img, "GPS Track Map", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img,
                f"Loop closed: {'YES' if local_status['loop_closed'] else 'NO'}",
                (20, map_h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    cv2.putText(img,
                f"Track length: {local_status['track_len_m']:.1f} m",
                (20, map_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)

    return img

# ============================================================
# GPS / TRACK HELPERS
# ============================================================

def latlon_to_xy(lat0, lon0, lat, lon):
    x = math.radians(lon - lon0) * R_EARTH * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * R_EARTH
    return x, y


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def unwrap_angle_diff(a):
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


class GPSTrackModel:
    def __init__(self):
        self.origin_lat = None
        self.origin_lon = None
        self.filtered_points = []
        self.last_committed_xy = None
        self.track_length_m = 0.0

        self.loop_closed = False
        self.loop_points = None
        self.loop_cumdist = None
        self.loop_length_m = 0.0
        self.turns = []
        self.current_xy = None

    def update_status(self, text, lat=None, lon=None, sats=None, hdop=None):
        with hud_lock:
            gps_status["status_text"] = text
            if lat is not None:
                gps_status["lat"] = round(lat, 6)
            if lon is not None:
                gps_status["lon"] = round(lon, 6)
            gps_status["sats"] = sats
            gps_status["hdop"] = hdop
            gps_status["loop_closed"] = self.loop_closed
            gps_status["track_len_m"] = self.track_length_m
            gps_status["loop_len_m"] = self.loop_length_m

    def process_fix(self, lat, lon, sats, hdop):
        if self.origin_lat is None:
            self.origin_lat = lat
            self.origin_lon = lon

        x, y = latlon_to_xy(self.origin_lat, self.origin_lon, lat, lon)
        current_xy = (x, y)
        self.current_xy = current_xy

        self.update_status("Tracking path...", lat, lon, sats, hdop)

        if self.last_committed_xy is None:
            self.filtered_points.append(current_xy)
            self.last_committed_xy = current_xy
            with gps_lock:
                points_xy.clear()
                points_xy.append(current_xy)
            return

        move_dist = dist(current_xy, self.last_committed_xy)
        if move_dist < GPS_MIN_UPDATE_DIST_M:
            if self.loop_closed:
                self.update_guidance_only()
            return

        # Freeze map/track learning once loop is closed.
        if self.loop_closed:
            self.current_xy = current_xy
            self.update_guidance_only()
            return

        self.filtered_points.append(current_xy)
        self.track_length_m += move_dist
        self.last_committed_xy = current_xy

        with gps_lock:
            points_xy.append(current_xy)

        if (not self.loop_closed) and self.should_close_loop():
            if self.build_closed_loop_model():
                self.loop_closed = True
                self.update_status("Loop learned. Live turn guidance active.", lat, lon, sats, hdop)
                self.update_guidance_only()
            else:
                self.update_status("Loop closure candidate found, refining...", lat, lon, sats, hdop)

        if self.loop_closed:
            self.update_guidance_only()
        else:
            clear_turn()
            self.update_status("Recording track until loop closes...", lat, lon, sats, hdop)

    def should_close_loop(self):
        if len(self.filtered_points) < GPS_MIN_LOOP_POINTS:
            return False
        if self.track_length_m < GPS_MIN_LOOP_LENGTH_M:
            return False
        return dist(self.filtered_points[0], self.filtered_points[-1]) <= GPS_LOOP_CLOSE_DIST_M

    # Reverted to the previous GPS loop-building method that was working better.
    def build_closed_loop_model(self):
        pts = np.array(self.filtered_points, dtype=np.float32)
        if len(pts) < GPS_MIN_LOOP_POINTS:
            return False

        if np.linalg.norm(pts[0] - pts[-1]) <= GPS_LOOP_CLOSE_DIST_M:
            pts[-1] = pts[0]
        elif np.linalg.norm(pts[0] - pts[-1]) > 1e-3:
            pts = np.vstack([pts, pts[0]])

        segs = pts[1:] - pts[:-1]
        seg_lens = np.linalg.norm(segs, axis=1)
        if np.sum(seg_lens) < GPS_MIN_LOOP_LENGTH_M:
            return False

        cum = np.zeros(len(pts), dtype=np.float32)
        cum[1:] = np.cumsum(seg_lens)
        headings = np.arctan2(segs[:, 1], segs[:, 0])

        turns = []
        last_turn_dist = -1e9
        thresh = math.radians(GPS_TURN_ANGLE_DEG)

        for i in range(1, len(headings)):
            dtheta = unwrap_angle_diff(headings[i] - headings[i - 1])
            here = float(cum[i])
            if abs(dtheta) >= thresh and (here - last_turn_dist) >= GPS_MIN_TURN_SPACING_M:
                turns.append({
                    "idx": i,
                    "distance_m": here,
                    "direction": "LEFT" if dtheta > 0 else "RIGHT"
                })
                last_turn_dist = here

        if len(turns) < 3:
            return False

        self.loop_points = pts
        self.loop_cumdist = cum
        self.loop_length_m = float(cum[-1])
        self.turns = turns
        return True

    def project_onto_loop(self, xy):
        pts = self.loop_points
        cum = self.loop_cumdist

        best_d2 = None
        best_progress = 0.0

        for i in range(len(pts) - 1):
            a = pts[i]
            b = pts[i + 1]
            ab = b - a
            ab_len2 = float(np.dot(ab, ab))
            if ab_len2 < 1e-9:
                continue

            ap = np.array([xy[0] - a[0], xy[1] - a[1]], dtype=np.float32)
            t = float(np.dot(ap, ab) / ab_len2)
            t = max(0.0, min(1.0, t))
            proj = a + t * ab

            dx = xy[0] - float(proj[0])
            dy = xy[1] - float(proj[1])
            d2 = dx * dx + dy * dy

            if best_d2 is None or d2 < best_d2:
                best_d2 = d2
                seg_len = float(cum[i + 1] - cum[i])
                best_progress = float(cum[i] + t * seg_len)

        return best_progress, math.sqrt(best_d2) if best_d2 is not None else 1e9

    def update_guidance_only(self):
        if not self.loop_closed or self.current_xy is None or not self.turns:
            clear_turn()
            return

        progress, offtrack = self.project_onto_loop(self.current_xy)
        if offtrack > GPS_MAX_MATCH_DIST_M:
            clear_turn()
            self.update_status("Off learned loop - no guidance.")
            return

        # Prevent a dead spot at the start/finish closure by allowing a small
        # slack region to wrap directly onto the first learned turn.
        wrap_progress = progress
        if self.loop_length_m > 0 and wrap_progress >= (self.loop_length_m - GPS_LOOP_PROGRESS_SLACK_M):
            wrap_progress = wrap_progress - self.loop_length_m

        next_turn = None
        for t in self.turns:
            if t["distance_m"] > wrap_progress + 0.5:
                next_turn = t
                break

        if next_turn is None:
            next_turn = self.turns[0]
            dist_to_turn = (self.loop_length_m - progress) + next_turn["distance_m"]
        else:
            if wrap_progress < 0.0:
                dist_to_turn = next_turn["distance_m"] - wrap_progress
            else:
                dist_to_turn = next_turn["distance_m"] - progress

        trigger_turn(next_turn["direction"], dist_to_turn)
        self.update_status(f"Guidance active - next turn {next_turn['direction']} in {dist_to_turn:.1f} m.")


track_model = GPSTrackModel()

# ============================================================
# GPS THREAD
# ============================================================

def open_serial():
    return serial.Serial(PORT, BAUD, timeout=1)


def read_one_fix(ser):
    while gps_running:
        raw = ser.readline().decode("ascii", errors="ignore").strip()
        if not raw.startswith("$"):
            continue

        try:
            msg = pynmea2.parse(raw)
        except Exception:
            continue

        if isinstance(msg, pynmea2.GGA):
            try:
                fix_quality = int(msg.gps_qual or 0)
            except ValueError:
                fix_quality = 0

            if fix_quality > 0 and msg.latitude and msg.longitude:
                return {
                    "lat": float(msg.latitude),
                    "lon": float(msg.longitude),
                    "sats": msg.num_sats,
                    "hdop": msg.horizontal_dil,
                }
    return None


def gps_reader_thread():
    try:
        ser = open_serial()
    except Exception as e:
        track_model.update_status(f"GPS serial open failed: {e}")
        return

    track_model.update_status(f"GPS opened on {PORT} @ {BAUD}")

    try:
        while gps_running:
            fix = read_one_fix(ser)
            if fix is None:
                continue
            track_model.process_fix(fix["lat"], fix["lon"], fix["sats"], fix["hdop"])
    except Exception as e:
        track_model.update_status(f"GPS thread error: {e}")
    finally:
        try:
            ser.close()
        except Exception:
            pass
        track_model.update_status("GPS serial closed")

# ============================================================
# EDGE DETECTION HELPERS
# ============================================================

def morphology_close(img_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel)


def morphology_open(img_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)


def make_main_roi_mask(shape_hw):
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)

    trap = np.array([[
        (int(ROI["left_bottom"] * w), int(ROI["bottom_y"] * h)),
        (int(ROI["left_top"] * w),    int(ROI["top_y"] * h)),
        (int(ROI["right_top"] * w),   int(ROI["top_y"] * h)),
        (int(ROI["right_bottom"] * w), int(ROI["bottom_y"] * h)),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, trap, 255)

    y_cut = int(BOTTOM_CUT_FRAC * h)
    mask[y_cut:, :] = 0

    if CAR_MASK["enabled"]:
        car_poly = np.array([[
            (int(CAR_MASK["left"] * w), h),
            (int(CAR_MASK["left"] * w), int(CAR_MASK["top_y"] * h)),
            (int(CAR_MASK["right"] * w), int(CAR_MASK["top_y"] * h)),
            (int(CAR_MASK["right"] * w), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, car_poly, 0)

    c0 = int(CENTER_BLOCK_L * w)
    c1 = int(CENTER_BLOCK_R * w)
    mask[:, c0:c1] = 0

    return mask


def build_paint_mask(frame_bgr, roi_mask):
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    white = cv2.inRange(hsv, (0, 0, WHITE_V_MIN), (179, WHITE_S_MAX, 255))

    if USE_YELLOW:
        yellow = cv2.inRange(hsv, YELLOW_LOW, YELLOW_HIGH)
        paint = cv2.bitwise_or(white, yellow)
    else:
        paint = white

    paint = cv2.GaussianBlur(paint, (5, 5), 0)
    paint = morphology_close(paint, 5)
    paint = morphology_open(paint, 3)
    paint = cv2.bitwise_and(paint, roi_mask)
    return paint


def thin_boundary_candidates(frame_bgr, paint_mask, roi_mask):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (BLUR_K, BLUR_K), 0)

    gradx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)

    pos = (gradx > GRAD_THRESH).astype(np.uint8) * 255
    neg = (gradx < -GRAD_THRESH).astype(np.uint8) * 255

    pos = cv2.bitwise_and(pos, paint_mask)
    neg = cv2.bitwise_and(neg, paint_mask)

    pos = cv2.bitwise_and(pos, roi_mask)
    neg = cv2.bitwise_and(neg, roi_mask)

    pos = morphology_open(pos, 3)
    neg = morphology_open(neg, 3)

    thin = cv2.bitwise_or(pos, neg)
    return thin


def make_debug_mask(paint_mask, thin_mask, roi_mask):
    h, w = paint_mask.shape
    dbg = np.zeros((h, w, 3), dtype=np.uint8)
    dbg[roi_mask > 0] = (15, 15, 15)
    dbg[paint_mask > 0] = (70, 70, 70)
    dbg[thin_mask > 0] = (255, 255, 255)
    return dbg


def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]
    roi_mask = make_main_roi_mask((h, w))
    paint_mask = build_paint_mask(frame_bgr, roi_mask)
    thin_mask = thin_boundary_candidates(frame_bgr, paint_mask, roi_mask)
    return make_debug_mask(paint_mask, thin_mask, roi_mask)

# ============================================================
# LIVE CAMERA READER THREAD
# ============================================================

def video_reader_thread():
    global video_running, video_fps, video_width, video_height

    cam = None

    try:
        cam = Picamera2(camera_num=CAMERA_NUM)

        config = cam.create_video_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"},
            controls={"FrameRate": CAMERA_FPS}
        )

        cam.configure(config)
        cam.start()
        time.sleep(2.0)

        video_fps = float(CAMERA_FPS)
        video_width = int(CAMERA_WIDTH)
        video_height = int(CAMERA_HEIGHT)

        frame_period = 1.0 / video_fps
        next_time = time.perf_counter()

        while video_running:
            frame_rgb = cam.capture_array()

            # Picamera2 gives RGB888 here; the existing edge pipeline expects BGR.
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            with frame_lock:
                if len(frame_queue) >= FRAME_BUFFER_SIZE:
                    frame_queue.popleft()
                frame_queue.append(frame)

            next_time += frame_period
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_time = time.perf_counter()

    except Exception as e:
        video_running = False
        with hud_lock:
            gps_status["status_text"] = f"Camera error: {e}"

    finally:
        if cam is not None:
            try:
                cam.stop()
            except Exception:
                pass

# ============================================================
# MAIN
# ============================================================

def main():
    global gps_running, video_running, video_fps, video_width, video_height

    gps_thread = threading.Thread(target=gps_reader_thread, daemon=True)
    gps_thread.start()

    reader_thread = threading.Thread(target=video_reader_thread, daemon=True)
    reader_thread.start()

    time.sleep(0.2)

    if video_width is None or video_height is None:
        video_width, video_height = CAMERA_WIDTH, CAMERA_HEIGHT

    out = None
    if WRITE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, video_fps, (video_width, video_height))
        if not out.isOpened():
            gps_running = False
            video_running = False
            raise RuntimeError(f"Could not open VideoWriter at: {OUTPUT_PATH}")

    if SHOW_ORIGINAL_VIDEO:
        cv2.namedWindow("Original Video", cv2.WINDOW_NORMAL)

    if SHOW_EDGE_VIDEO:
        cv2.namedWindow("Edge Detected Video", cv2.WINDOW_NORMAL)

    if SHOW_HUD:
        cv2.namedWindow("Racing HUD", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Racing HUD", DISPLAY_W, DISPLAY_H)

    if SHOW_GPS_MAP:
        cv2.namedWindow("GPS Map", cv2.WINDOW_NORMAL)

    hud_frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)

    proc_t0 = time.time()
    proc_count = 0

    try:
        while True:
            frame = None

            with frame_lock:
                if frame_queue:
                    frame = frame_queue.popleft()

            if frame is not None:
                edge_frame = process_frame(frame)

                if SHOW_ORIGINAL_VIDEO:
                    cv2.imshow("Original Video", frame)

                if SHOW_EDGE_VIDEO:
                    cv2.imshow("Edge Detected Video", edge_frame)

                if SHOW_HUD:
                    render_hud(hud_frame)
                    cv2.imshow("Racing HUD", hud_frame)

                if SHOW_GPS_MAP:
                    gps_map = render_gps_map()
                    cv2.imshow("GPS Map", gps_map)

                if out is not None:
                    out.write(edge_frame)

                proc_count += 1
                now = time.time()
                if now - proc_t0 >= 1.0:
                    print(f"Display/processing FPS: {proc_count / (now - proc_t0):.2f}")
                    proc_t0 = now
                    proc_count = 0

                frame_delay = max(1, int(round(1000.0 / max(video_fps, 1e-3))))
                key = cv2.waitKey(frame_delay) & 0xFF
            else:
                if SHOW_HUD:
                    render_hud(hud_frame)
                    cv2.imshow("Racing HUD", hud_frame)

                if SHOW_GPS_MAP:
                    gps_map = render_gps_map()
                    cv2.imshow("GPS Map", gps_map)

                key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

    finally:
        gps_running = False
        video_running = False
        time.sleep(0.2)

        if out is not None:
            out.release()
            print(f"Saved: {OUTPUT_PATH}")

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
