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
from PIL import Image, ImageDraw, ImageFont
from waveshare_OLED import OLED_1in51

# ============================================================
# GLOBAL CONFIG
# ============================================================

# ----------------------------
# GPS
# ----------------------------
PORT = "COM9"
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
# Video
# ----------------------------
PATH_OF_VID = "racing_vid.mp4"

WRITE_OUTPUT = False
OUTPUT_PATH = Path.home() / "local_work_tegra" / "track_edge_mask_with_scored_limits.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = str(OUTPUT_PATH)

# Keep edge detection identical; speed comes from display/scheduling only.
SHOW_ORIGINAL_VIDEO = False       # Original window costs time. Set True if needed.
SHOW_EDGE_VIDEO = True
SHOW_HUD = True                  # True = draw the HUD on the Waveshare 1.51 inch OLED/LCD instead of an OpenCV monitor window.
SHOW_GPS_MAP = False              # GPS map is expensive. Set True if needed.

# Real-time mode: keep only latest frame so the display never builds lag.
FRAME_BUFFER_SIZE = 1
GPS_MAP_UPDATE_EVERY = 10
HUD_UPDATE_EVERY = 2
PRINT_FPS = True

# ----------------------------
# Track position window
# ----------------------------
SHOW_TRACK_POSITION = True
POSITION_DISPLAY_W = 520
POSITION_DISPLAY_H = 260

# If top two edge scores are within this relative difference, call it CENTER.
# Example: 0.12 means scores within about 12 percent are treated as balanced.
POSITION_CENTER_REL_DIFF = 0.06

# Debounce: require the same raw position decision across several recent frames
# before changing the displayed position. This prevents 1-2 frame pixel glitches.
POSITION_HISTORY_FRAMES = 2
POSITION_CONFIRM_FRAMES = 2
MIN_HIGHLIGHT_LENGTH_PX = 600.0

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
    left=0,
    right=1,
    top_y=0.4, #0.42 works
)

BOTTOM_CUT_FRAC = 0.88

# ----------------------------
# Paint detection
# ----------------------------
USE_YELLOW = False

WHITE_V_MIN = 170
WHITE_S_MAX = 55

YELLOW_LOW = (12, 90, 120)
YELLOW_HIGH = (45, 255, 255)

# ----------------------------
# Preprocessing
# ----------------------------
BLUR_K = 5

# ----------------------------
# Thin-edge extraction
# ----------------------------
GRAD_THRESH = 26

CENTER_BLOCK_L = 0.43
CENTER_BLOCK_R = 0.43

# ============================================================
# TRACK LIMIT SCORING CONFIG
# ============================================================

# Higher = more lines highlighted. 2 is usually left/right track edge.
TRACK_LIMIT_TOP_N = 2

# Reject tiny edge fragments before scoring.
MIN_EDGE_AREA_PX = 25
MIN_EDGE_LENGTH_PX = 35.0
MIN_EDGE_HEIGHT_PX = 18.0

# Distance-from-camera score: image bottom is closest, so candidates with a
# lower edge nearer the bottom of ROI get boosted heavily.
DISTANCE_WEIGHT = 0.58
CONTINUITY_WEIGHT = 0.27
EDGE_STRENGTH_WEIGHT = 0.15

# Continuity tuning.
# Larger values give more reward to long, vertically spanning connected lines.
CONTINUITY_LENGTH_REF_FRAC = 0.45
CONTINUITY_HEIGHT_REF_FRAC = 0.45

# Fit / straightness reward. Track-limit paint can curve, so this is gentle.
LINE_FIT_WEIGHT = 0.20

# Draw colors in BGR.
COL_SCORE_1 = (0, 0, 255)       # highest rank: red
COL_SCORE_2 = (0, 165, 255)     # second: orange
COL_SCORE_OTHER = (255, 0, 255) # optional lower ranks: magenta
COL_CENTROID = (255, 255, 255)

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

# Cached ROI mask does not alter detection output for same frame size; it only avoids rebuilding same mask.
_roi_mask_cache = {}

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


# ============================================================
# OLED / LCD HUD HELPERS
# This replaces the old OpenCV "Racing HUD" monitor window.
# Edge-video, track-position, and GPS-map windows are left unchanged.
# ============================================================

OLED_ROTATE_180 = True
OLED_DOT_BLINK_S = 0.5


def draw_oled_right_arrow(draw, cx, cy):
    # Same thick-arrow style as your working LCD test code.
    draw.rectangle((cx - 40, cy - 6, cx + 15, cy + 6), fill=0)
    draw.polygon([
        (cx + 15, cy - 22),
        (cx + 45, cy),
        (cx + 15, cy + 22),
    ], fill=0)


def draw_oled_left_arrow(draw, cx, cy):
    # Same thick-arrow style as your working LCD test code.
    draw.rectangle((cx - 15, cy - 6, cx + 40, cy + 6), fill=0)
    draw.polygon([
        (cx - 15, cy - 22),
        (cx - 45, cy),
        (cx - 15, cy + 22),
    ], fill=0)


def draw_oled_blink_dot(draw, width, dot_on):
    if dot_on:
        draw.ellipse((width - 18, 4, width - 4, 18), fill=0)


class OLEDHUD:
    def __init__(self):
        self.disp = OLED_1in51.OLED_1in51()
        self.disp.Init()
        self.disp.clear()
        self.width = self.disp.width
        self.height = self.disp.height
        self.cx = self.width // 2
        self.cy = self.height // 2
        self.font_small = ImageFont.load_default()
        self.font_big = ImageFont.load_default()
        self.dot_on = True
        self.last_dot_switch = time.time()

    def _text_center(self, draw, text, y, font=None, fill=0):
        font = font or self.font_small
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(text) * 6
        draw.text(((self.width - tw) // 2, y), text, font=font, fill=fill)

    def render(self):
        now = time.time()
        if now - self.last_dot_switch >= OLED_DOT_BLINK_S:
            self.dot_on = not self.dot_on
            self.last_dot_switch = now

        with hud_lock:
            local_turn = turn_dir
            local_dist = distance_m
            local_active = active
            local_status = gps_status.copy()

        image = Image.new("1", (self.width, self.height), "WHITE")
        draw = ImageDraw.Draw(image)

        if local_active and local_turn:
            # Original monitor HUD: arrow + TURN LEFT/RIGHT + distance.
            if local_turn == "LEFT":
                draw_oled_left_arrow(draw, self.cx, self.cy - 10)
            else:
                draw_oled_right_arrow(draw, self.cx, self.cy - 10)

            self._text_center(draw, f"TURN {local_turn}", 2)
            self._text_center(draw, f"{max(local_dist, 0):.0f} m", self.height - 18)
        else:
            # Original monitor HUD idle state.
            self._text_center(draw, "TRACK CLEAR", self.cy - 8)

        # The monitor HUD also showed GPS status. On the small OLED this is
        # reduced to a compact bottom/top line so the arrow remains readable.
        if not (local_active and local_turn):
            status = str(local_status.get("status_text", ""))[:22]
            self._text_center(draw, status, self.height - 18)

        draw_oled_blink_dot(draw, self.width, self.dot_on)

        if OLED_ROTATE_180:
            image = image.rotate(180)

        self.disp.ShowImage(self.disp.getbuffer(image))

    def close(self):
        try:
            self.disp.clear()
            self.disp.module_exit()
        except Exception:
            pass


def render_hud(frame):
    """
    Kept only so the rest of your file does not need to change.
    The OLED HUD is rendered by OLEDHUD.render(); this OpenCV frame renderer is
    no longer used when SHOW_HUD=True.
    """
    frame[:] = COL_BG

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
# These are intentionally kept the same as your working version.
# ============================================================

def morphology_close(img_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel)


def morphology_open(img_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)


def make_main_roi_mask(shape_hw):
    h, w = shape_hw
    cache_key = (h, w)
    cached = _roi_mask_cache.get(cache_key)
    if cached is not None:
        return cached

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
            (int(CAR_MASK["left"] * w), h),                      # bottom-left
            (int(CAR_MASK["left"] * w), int(CAR_MASK["top_y"] * h)),  # top-left
            (int(CAR_MASK["right"] * w), int(CAR_MASK["top_y"] * h)), # top-right
            (int(CAR_MASK["right"] * w), h),                     # bottom-right
        ]], dtype=np.int32)

        cv2.fillPoly(mask, car_poly, 0)

    c0 = int(CENTER_BLOCK_L * w)
    c1 = int(CENTER_BLOCK_R * w)
    mask[:, c0:c1] = 0

    _roi_mask_cache[cache_key] = mask
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
    grad_strength = cv2.convertScaleAbs(np.abs(gradx))
    grad_strength = cv2.bitwise_and(grad_strength, roi_mask)
    return thin, grad_strength


def make_debug_mask(paint_mask, thin_mask, roi_mask):
    h, w = paint_mask.shape
    dbg = np.zeros((h, w, 3), dtype=np.uint8)
    dbg[roi_mask > 0] = (15, 15, 15)
    dbg[paint_mask > 0] = (70, 70, 70)
    dbg[thin_mask > 0] = (255, 255, 255)
    return dbg

# ============================================================
# TRACK LIMIT SCORING HELPERS
# These are intentionally kept the same as your working version.
# ============================================================

def contour_polyline_length(contour):
    if contour is None or len(contour) < 2:
        return 0.0
    return float(cv2.arcLength(contour, False))


def contour_line_fit_score(points_xy):
    if points_xy is None or len(points_xy) < 6:
        return 0.0

    pts = points_xy.reshape(-1, 2).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    direction = np.array([vx, vy], dtype=np.float32)
    origin = np.array([x0, y0], dtype=np.float32)
    rel = pts - origin
    projection = rel @ direction
    closest = origin + np.outer(projection, direction)
    errors = np.linalg.norm(pts - closest, axis=1)
    rms = float(np.sqrt(np.mean(errors * errors)))

    # Convert RMS error to 0..1. A curved line still receives partial credit.
    return 1.0 / (1.0 + rms / 12.0)


def score_track_limit_edges(thin_mask, grad_strength, roi_mask):
    h, w = thin_mask.shape[:2]

    # Close small gaps before connected-component/contour scoring. This makes
    # dashed or partly broken paint lines rank as one candidate when close.
    candidate_mask = morphology_close(thin_mask, 5)
    candidate_mask = cv2.dilate(candidate_mask, np.ones((3, 3), np.uint8), iterations=1)
    candidate_mask = cv2.bitwise_and(candidate_mask, roi_mask)

    contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    scored = []
    roi_ys = np.where(roi_mask > 0)[0]
    if len(roi_ys) > 0:
        roi_top = int(roi_ys.min())
        roi_bottom = int(roi_ys.max())
    else:
        roi_top = 0
        roi_bottom = h - 1

    roi_height = max(1, roi_bottom - roi_top)
    length_ref = max(1.0, CONTINUITY_LENGTH_REF_FRAC * math.hypot(w, h))
    height_ref = max(1.0, CONTINUITY_HEIGHT_REF_FRAC * roi_height)

    for contour in contours:
        area = float(cv2.contourArea(contour))
        x, y, bw, bh = cv2.boundingRect(contour)
        length = contour_polyline_length(contour)

        if area < MIN_EDGE_AREA_PX and length < MIN_EDGE_LENGTH_PX:
            continue
        if bh < MIN_EDGE_HEIGHT_PX:
            continue

        component_mask = np.zeros_like(thin_mask)
        cv2.drawContours(component_mask, [contour], -1, 255, thickness=cv2.FILLED)
        component_thin = cv2.bitwise_and(thin_mask, component_mask)
        ys, xs = np.where(component_thin > 0)
        if len(xs) < MIN_EDGE_LENGTH_PX:
            continue

        # Distance from camera: bottom-most actual thin edge pixel is closest.
        bottom_y = float(np.max(ys))
        distance_score = (bottom_y - roi_top) / float(roi_height)
        distance_score = max(0.0, min(1.0, distance_score))

        # Continuity: long + vertically spanning + line-fit consistency.
        thin_count = float(len(xs))
        if thin_count < MIN_HIGHLIGHT_LENGTH_PX:
            continue
        length_score = min(1.0, thin_count / length_ref)
        height_score = min(1.0, float(np.max(ys) - np.min(ys) + 1) / height_ref)
        fit_score = contour_line_fit_score(contour)
        continuity_score = ((1.0 - LINE_FIT_WEIGHT) * (0.55 * length_score + 0.45 * height_score)
                            + LINE_FIT_WEIGHT * fit_score)

        edge_vals = grad_strength[component_thin > 0]
        if edge_vals.size > 0:
            # 255 roughly means very strong Scharr response after convertScaleAbs.
            strength_score = min(1.0, float(np.mean(edge_vals)) / 255.0)
        else:
            strength_score = 0.0

        total_score = (
            DISTANCE_WEIGHT * distance_score +
            CONTINUITY_WEIGHT * continuity_score +
            EDGE_STRENGTH_WEIGHT * strength_score
        )

        scored.append({
            "score": total_score,
            "distance_score": distance_score,
            "continuity_score": continuity_score,
            "strength_score": strength_score,
            "contour": contour,
            "thin_mask": component_thin,
            "bbox": (x, y, bw, bh),
            "bottom_y": bottom_y,
            "thin_count": thin_count,
        })

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored


def overlay_ranked_track_limits(debug_bgr, scored_edges, top_n=TRACK_LIMIT_TOP_N):
    out = debug_bgr.copy()

    for rank, item in enumerate(scored_edges[:top_n], start=1):
        if rank == 1:
            color = COL_SCORE_1
            thickness = 4
        elif rank == 2:
            color = COL_SCORE_2
            thickness = 3
        else:
            color = COL_SCORE_OTHER
            thickness = 2

        # Draw the contour outline and brighten the actual thin pixels.
        cv2.drawContours(out, [item["contour"]], -1, color, thickness, cv2.LINE_AA)
        out[item["thin_mask"] > 0] = color

        x, y, bw, bh = item["bbox"]
        label = (f"#{rank} score {item['score']:.2f}  "
                 f"D {item['distance_score']:.2f} C {item['continuity_score']:.2f} "
                 f"E {item['strength_score']:.2f}")
        text_y = max(22, y - 8)
        cv2.putText(out, label, (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

        moments = cv2.moments(item["contour"])
        if abs(moments["m00"]) > 1e-6:
            cx = int(moments["m10"] / moments["m00"])
            cy = int(moments["m01"] / moments["m00"])
            cv2.circle(out, (cx, cy), 4, COL_CENTROID, -1, cv2.LINE_AA)

    cv2.putText(out, "Ranked track-limit candidates: distance > continuity > edge strength",
                (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    return out


def process_frame(frame_bgr):
    # Full-resolution, exact same edge pipeline. No resizing. No interpolation. No threshold changes.
    # The only addition is returning scored_edges so the position window can read rank #1/#2.
    h, w = frame_bgr.shape[:2]
    roi_mask = make_main_roi_mask((h, w))
    paint_mask = build_paint_mask(frame_bgr, roi_mask)
    thin_mask, grad_strength = thin_boundary_candidates(frame_bgr, paint_mask, roi_mask)
    debug = make_debug_mask(paint_mask, thin_mask, roi_mask)
    scored_edges = score_track_limit_edges(thin_mask, grad_strength, roi_mask)
    edge_frame = overlay_ranked_track_limits(debug, scored_edges, TRACK_LIMIT_TOP_N)
    return edge_frame, scored_edges


# ============================================================
# TRACK POSITION HELPERS
# ============================================================

class TrackPositionEstimator:
    def __init__(self, history_frames=POSITION_HISTORY_FRAMES, confirm_frames=POSITION_CONFIRM_FRAMES):
        self.history = deque(maxlen=max(1, int(history_frames)))
        self.confirm_frames = max(1, int(confirm_frames))
        self.stable_position = "WAITING"
        self.raw_position = "WAITING"
        self.last_info = {}

    @staticmethod
    def edge_center_x(edge_item):
        x, y, bw, bh = edge_item["bbox"]
        moments = cv2.moments(edge_item["contour"])
        if abs(moments["m00"]) > 1e-6:
            return float(moments["m10"] / moments["m00"])
        return float(x + 0.5 * bw)

    def update(self, scored_edges):
        if len(scored_edges) < 2:
            raw = "WAITING"
            info = {"reason": "Need two confirmed edge candidates"}
        else:
            edge1 = scored_edges[0]
            edge2 = scored_edges[1]

            score1 = float(edge1["score"])
            score2 = float(edge2["score"])
            max_score = max(score1, score2, 1e-6)
            rel_diff = abs(score1 - score2) / max_score

            x1 = self.edge_center_x(edge1)
            x2 = self.edge_center_x(edge2)

            if rel_diff <= POSITION_CENTER_REL_DIFF:
                raw = "CENTER"
            elif x1 < x2:
                raw = "LEFT"
            else:
                raw = "RIGHT"

            info = {
                "rank1_x": x1,
                "rank2_x": x2,
                "rank1_score": score1,
                "rank2_score": score2,
                "relative_diff": rel_diff,
                "reason": "Scores close" if raw == "CENTER" else "Rank #1 edge side",
            }

        self.raw_position = raw
        self.last_info = info
        self.history.append(raw)

        # Only change the stable display when the same raw result appears enough
        # times in recent history. This suppresses single-frame false edge glitches.
        for candidate in ("LEFT", "RIGHT", "CENTER", "WAITING"):
            if sum(1 for item in self.history if item == candidate) >= self.confirm_frames:
                self.stable_position = candidate
                break

        return self.stable_position


def render_track_position(frame, estimator):
    frame[:] = COL_BG
    H, W = frame.shape[:2]

    position = estimator.stable_position
    raw = estimator.raw_position
    info = estimator.last_info

    if position == "LEFT":
        col = COL_RED
        label = "CAR LEFT"
        sub = "Highest edge is left of second edge"
    elif position == "RIGHT":
        col = COL_YELLOW
        label = "CAR RIGHT"
        sub = "Highest edge is right of second edge"
    elif position == "CENTER":
        col = COL_GREEN
        label = "CAR CENTER"
        sub = "Top two edge scores are close"
    else:
        col = COL_WHITE
        label = "WAITING"
        sub = "Need stable top two edge candidates"

    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.45, 3)
    cv2.putText(frame, label, ((W - tw) // 2, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 1.45, col, 3, cv2.LINE_AA)

    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)
    cv2.putText(frame, sub, ((W - sw) // 2, 112),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, COL_WHITE, 1, cv2.LINE_AA)

    cv2.putText(frame, f"Raw: {raw}   Stable: {position}", (28, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, COL_WHITE, 2, cv2.LINE_AA)

    if "rank1_score" in info and "rank2_score" in info:
        cv2.putText(frame, f"#1 score: {info['rank1_score']:.3f}   x: {info['rank1_x']:.0f}", (28, 194),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COL_SCORE_1, 2, cv2.LINE_AA)
        cv2.putText(frame, f"#2 score: {info['rank2_score']:.3f}   x: {info['rank2_x']:.0f}", (28, 224),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COL_SCORE_2, 2, cv2.LINE_AA)
        cv2.putText(frame, f"Score diff: {100.0 * info['relative_diff']:.1f}%   Center threshold: {100.0 * POSITION_CENTER_REL_DIFF:.1f}%", (28, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, COL_WHITE, 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, info.get("reason", "No edge info yet"), (28, 205),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, COL_WHITE, 2, cv2.LINE_AA)

# ============================================================
# VIDEO READER THREAD
# ============================================================

def video_reader_thread():
    global video_running, video_fps, video_width, video_height

    cap = cv2.VideoCapture(PATH_OF_VID)
    if not cap.isOpened():
        video_running = False
        return

    # Keep latency low on camera-like backends when supported.
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 1e-3:
        video_fps = fps
    else:
        video_fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        width, height = 640, 480
    video_width = width
    video_height = height

    frame_period = 1.0 / video_fps
    next_time = time.perf_counter()

    while video_running:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            cap = cv2.VideoCapture(PATH_OF_VID)
            if not cap.isOpened():
                video_running = False
                break

            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 1e-3:
                video_fps = fps
            else:
                video_fps = 30.0

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if width > 0 and height > 0:
                video_width = width
                video_height = height

            frame_period = 1.0 / video_fps
            next_time = time.perf_counter()
            continue

        # Real-time behavior: always replace queued frame with newest frame.
        # This does not change edge detection quality; it only prevents old-frame lag.
        with frame_lock:
            frame_queue.clear()
            frame_queue.append(frame)

        next_time += frame_period
        sleep_time = next_time - time.perf_counter()
        if sleep_time > 0:
            time.sleep(sleep_time)
        else:
            next_time = time.perf_counter()

    cap.release()

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
        video_width, video_height = 640, 480

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

    oled_hud = None
    if SHOW_HUD:
        try:
            oled_hud = OLEDHUD()
        except Exception as e:
            print(f"OLED HUD init failed: {e}")
            oled_hud = None

    if SHOW_GPS_MAP:
        cv2.namedWindow("GPS Map", cv2.WINDOW_NORMAL)

    if SHOW_TRACK_POSITION:
        cv2.namedWindow("Track Position", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Track Position", POSITION_DISPLAY_W, POSITION_DISPLAY_H)

    hud_frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    position_frame = np.zeros((POSITION_DISPLAY_H, POSITION_DISPLAY_W, 3), dtype=np.uint8)
    position_estimator = TrackPositionEstimator()
    gps_map = None
    loop_count = 0

    proc_t0 = time.time()
    proc_count = 0

    try:
        while True:
            frame = None

            with frame_lock:
                if frame_queue:
                    # Take the newest frame and drop anything older.
                    frame = frame_queue.pop()
                    frame_queue.clear()

            if frame is not None:
                edge_frame, scored_edges = process_frame(frame)

                if SHOW_ORIGINAL_VIDEO:
                    cv2.imshow("Original Video", frame)

                if SHOW_EDGE_VIDEO:
                    cv2.imshow("Edge Detected Video", edge_frame)

                if SHOW_TRACK_POSITION:
                    position_estimator.update(scored_edges)
                    render_track_position(position_frame, position_estimator)
                    cv2.imshow("Track Position", position_frame)

                loop_count += 1

                if SHOW_HUD and oled_hud is not None and (loop_count % HUD_UPDATE_EVERY == 0):
                    oled_hud.render()

                if SHOW_GPS_MAP:
                    if gps_map is None or loop_count % GPS_MAP_UPDATE_EVERY == 0:
                        gps_map = render_gps_map()
                    cv2.imshow("GPS Map", gps_map)

                if out is not None:
                    out.write(edge_frame)

                proc_count += 1
                now = time.time()
                if PRINT_FPS and now - proc_t0 >= 1.0:
                    print(f"Display/processing FPS: {proc_count / (now - proc_t0):.2f}")
                    proc_t0 = now
                    proc_count = 0

                # Do not add video-frame delay here; reader thread already paces playback.
                key = cv2.waitKey(1) & 0xFF
            else:
                if SHOW_HUD and oled_hud is not None:
                    oled_hud.render()

                if SHOW_GPS_MAP:
                    if gps_map is None:
                        gps_map = render_gps_map()
                    cv2.imshow("GPS Map", gps_map)

                if SHOW_TRACK_POSITION:
                    cv2.imshow("Track Position", position_frame)

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

        if 'oled_hud' in locals() and oled_hud is not None:
            oled_hud.close()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
