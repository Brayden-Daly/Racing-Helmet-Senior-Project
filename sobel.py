#!/usr/bin/env python3
import time
from pathlib import Path

import cv2
import numpy as np

# ----------------------------
# Config
# ----------------------------
PATH_OF_VID = "racing_vid.mp4"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

WRITE_OUTPUT = False
OUTPUT_PATH = Path.home() / "local_work_tegra" / "track_hough_output.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = str(OUTPUT_PATH)

# ROI trapezoid (tune as needed)
ROI = dict(
    keep_trapezoid=True,
    cut_car=False,

    top_y=0.17,
    bottom_y=0.40,

    left_bottom=0.05,
    right_bottom=0.95,
    left_top=0.25,
    right_top=0.75,
)

# Edge detection
CANNY_SIGMA = 0.25
EDGE_BIN_THRESH = 2
DO_CLOSE = True

# Hough params (tune)
HOUGH_THRESHOLD = 35
MIN_LINE_LENGTH = 55
MAX_LINE_GAP = 55

# Segment filtering / scoring
SLOPE_MIN = 0.30
LEFT_X_MAX = 0.55
RIGHT_X_MIN = 0.45
SUPPORT_WEIGHT = 0.6

# Draw
TOP_K_SEGMENTS = 10
SEG_THICKNESS = 6

# Target point (guidance cue)
Y_TARGET_FRAC = 0.48  # will be clamped to ROI

# --- Preprocessing upgrades ---
USE_LAB_L = True
USE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)

ILLUM_NORM = True
ILLUM_BLUR_K = 31  # odd, bigger = more shadow removal (try 21..61)

USE_BILATERAL = False  # set True if you can afford CPU
BILATERAL_D = 7
BILATERAL_SIGMA_COLOR = 60
BILATERAL_SIGMA_SPACE = 60

# --- Post-edge cleanup ---
DO_OPEN = False
OPEN_K = 3

KEEP_LARGEST_CC = False
KEEP_CC_COUNT = 8          # keep top N connected components by area
KEEP_CC_MIN_AREA = 120     # reject tiny blobs no matter what

# Optional: reject components that are too "round" (helps remove signs)
CC_MIN_ASPECT = 2.0        # width/height or height/width must exceed this

# ----------------------------
# Helpers
# ----------------------------
def lab_l_channel(frame_bgr):
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    return L

def apply_clahe(gray_u8, clip=2.0, grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return clahe.apply(gray_u8)

def illum_normalize(gray_u8, blur_k=31):
    # Normalize illumination by dividing by a heavily blurred version (shadow suppression).
    k = int(blur_k)
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(gray_u8, (k, k), 0)
    blur = np.maximum(blur, 1)  # avoid divide-by-zero
    norm = (gray_u8.astype(np.float32) * 255.0) / blur.astype(np.float32)
    norm = np.clip(norm, 0, 255).astype(np.uint8)
    return norm

def morphology_open(img_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_OPEN, kernel)

def keep_largest_connected_components(bw_u8, keep_n=8, min_area=120, min_aspect=2.0):
    # bw_u8 is 0/255
    num, labels, stats, _ = cv2.connectedComponentsWithStats((bw_u8 > 0).astype(np.uint8), connectivity=8)
    if num <= 1:
        return bw_u8

    comps = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        aspect = (w / max(h, 1)) if w >= h else (h / max(w, 1))
        if aspect < min_aspect:
            continue
        comps.append((area, i))

    if not comps:
        return np.zeros_like(bw_u8)

    comps.sort(reverse=True, key=lambda t: t[0])
    keep_ids = set([i for _, i in comps[:keep_n]])

    out = np.zeros_like(bw_u8)
    for i in keep_ids:
        out[labels == i] = 255
    return out

def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def auto_canny(img, sigma=CANNY_SIGMA, min_thresh=10, max_thresh=180):
    v = np.median(img)
    lower = int(max(min_thresh, (1.0 - sigma) * v))
    upper = int(min(max_thresh, (1.0 + sigma) * v))
    upper = max(upper, lower + 20)
    return cv2.Canny(img, lower, upper)


def morphology_close(img_u8):
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(img_u8, cv2.MORPH_CLOSE, kernel)


def apply_roi_mask(img, keep_trapezoid=True, cut_car=False,
                   top_y=0.55, bottom_y=0.98,
                   left_bottom=0.08, right_bottom=0.92,
                   left_top=0.44, right_top=0.56,
                   car_cut_y=0.72,
                   car_left=0.18, car_right=0.82):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if keep_trapezoid:
        trap = np.array([[
            (int(left_bottom * w), int(bottom_y * h)),
            (int(left_top * w),    int(top_y * h)),
            (int(right_top * w),   int(top_y * h)),
            (int(right_bottom * w),int(bottom_y * h)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, trap, 255)
    else:
        mask[:] = 255

    if cut_car:
        car_poly = np.array([[
            (int(car_left * w),  int(1.00 * h)),
            (int(car_left * w),  int(car_cut_y * h)),
            (int(car_right * w), int(car_cut_y * h)),
            (int(car_right * w), int(1.00 * h)),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, car_poly, 0)

    return cv2.bitwise_and(img, img, mask=mask)


def draw_roi_outline(frame_bgr, roi):
    h, w = frame_bgr.shape[:2]
    out = frame_bgr.copy()
    trap = np.array([[
        (int(roi["left_bottom"] * w), int(roi["bottom_y"] * h)),
        (int(roi["left_top"]    * w), int(roi["top_y"]    * h)),
        (int(roi["right_top"]   * w), int(roi["top_y"]    * h)),
        (int(roi["right_bottom"]* w), int(roi["bottom_y"] * h)),
    ]], dtype=np.int32)
    cv2.polylines(out, trap, isClosed=True, color=(0, 255, 255), thickness=2)
    return out


def hough_lines_p(edge_u8, threshold=50, min_line_length=60, max_line_gap=30):
    return cv2.HoughLinesP(
        edge_u8, 1, np.pi/180, threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )


def segment_support_score(edge_bw, x1, y1, x2, y2, thickness=7):
    seg_mask = np.zeros_like(edge_bw)
    cv2.line(seg_mask, (x1, y1), (x2, y2), 255, thickness)
    return int(np.count_nonzero(cv2.bitwise_and(edge_bw, seg_mask)))


def split_score_segments(lines, edge_bw, w,
                         slope_min=SLOPE_MIN,
                         left_x_max=LEFT_X_MAX,
                         right_x_min=RIGHT_X_MIN,
                         support_weight=SUPPORT_WEIGHT):
    left_scored = []
    right_scored = []
    if lines is None:
        return left_scored, right_scored

    for x1, y1, x2, y2 in lines[:, 0]:
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 5:
            continue

        slope = dy / dx
        if abs(slope) < slope_min:
            continue

        length = (dx * dx + dy * dy) ** 0.5
        if length < 40:
            continue

        mx = 0.5 * (x1 + x2)
        support = segment_support_score(edge_bw, x1, y1, x2, y2, thickness=7)

        score = length + support_weight * support

        if slope < 0 and mx < left_x_max * w:
            left_scored.append((score, (x1, y1, x2, y2)))
        elif slope > 0 and mx > right_x_min * w:
            right_scored.append((score, (x1, y1, x2, y2)))

    left_scored.sort(key=lambda t: t[0], reverse=True)
    right_scored.sort(key=lambda t: t[0], reverse=True)
    return left_scored, right_scored


def draw_top_k_segments(img, scored_list, k=6, color=(0, 0, 255), thickness=5):
    for score, (x1, y1, x2, y2) in scored_list[:k]:
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def x_at_y(seg, yq):
    x1, y1, x2, y2 = seg
    if y2 == y1:
        return 0.5 * (x1 + x2)
    t = (yq - y1) / (y2 - y1)
    return x1 + t * (x2 - x1)


# ----------------------------
# Pipeline
# ----------------------------
def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]

    # --- ROI first (on color image) so we don't pick up stands/signs ---
    roi_frame = apply_roi_mask(frame_bgr, **ROI)

    # --- Paint mask (white + optional yellow) ---
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # White paint: low saturation, high value
    white = cv2.inRange(hsv, (0, 0, 170), (179, 60, 255))

    # Optional: yellow paint (enable if your track uses yellow)
    yellow = cv2.inRange(hsv, (15, 80, 120), (40, 255, 255))

    paint_mask = cv2.bitwise_or(white, yellow)

    # --- Clean the mask (connect strokes, remove specks) ---
    paint_mask = cv2.GaussianBlur(paint_mask, (5, 5), 0)

    # Close then open: close connects dashed/broken paint, open removes tiny noise
    if DO_CLOSE:
        paint_mask = morphology_close(paint_mask)
    if DO_OPEN:
        paint_mask = morphology_open(paint_mask, OPEN_K)

    # IMPORTANT: at this point, paint_mask is already your "bw"
    bw = paint_mask

    # --- Edges on the mask (optional but helps Hough) ---
    edges = cv2.Canny(bw, 50, 150)

    # If you want Hough on edges:
    hough_input = edges

    # Or if you want Hough on the filled mask:
    # hough_input = bw

    lines = hough_lines_p(
        hough_input,
        threshold=HOUGH_THRESHOLD,
        min_line_length=MIN_LINE_LENGTH,
        max_line_gap=MAX_LINE_GAP
    )

    left_scored, right_scored = split_score_segments(lines, (hough_input > 0).astype(np.uint8) * 255, w)

    output = frame_bgr.copy()
    draw_top_k_segments(output, left_scored,  k=TOP_K_SEGMENTS, color=(0, 0, 255), thickness=SEG_THICKNESS)
    draw_top_k_segments(output, right_scored, k=TOP_K_SEGMENTS, color=(0, 0, 255), thickness=SEG_THICKNESS)

    # Guidance cue (use best segment each side)
    y_top = int(ROI["top_y"] * h)
    y_bot = int(ROI["bottom_y"] * h)
    y_target = int(Y_TARGET_FRAC * h)
    y_target = max(y_top, min(y_bot, y_target))

    if len(left_scored) > 0 and len(right_scored) > 0:
        lx = x_at_y(left_scored[0][1], y_target)
        rx = x_at_y(right_scored[0][1], y_target)
        x_mid = int(np.clip(0.5 * (lx + rx), 0, w - 1))

        cv2.circle(output, (x_mid, y_target), 9, (0, 255, 255), -1)
        cv2.arrowedLine(output, (w // 2, y_bot), (x_mid, y_target),
                        (0, 255, 255), 3, tipLength=0.25)

    # Debug view: show bw so you can confirm paint is being detected
    return output, bw


# ----------------------------
# Main loop
# ----------------------------
def main():
    #get the video
    cap = cv2.VideoCapture(PATH_OF_VID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {PATH_OF_VID}")

    #get FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    out = None
    if WRITE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))
        if not out.isOpened():
            raise RuntimeError(f"Could not open VideoWriter at: {OUTPUT_PATH}")

    t0 = time.time()
    frames = 0

    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #resize the frame with frame width and frame height
        frame = resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)

        #get edges and output of frames and draw trapezoid
        output, edges_dbg = process_frame(frame)
        roi_dbg = draw_roi_outline(frame, ROI)

        #show outputs of edge detection, output and roi
        cv2.imshow("Apex overlay (q/ESC to quit)", output)
        cv2.imshow("Edges (ROI-masked)", edges_dbg)
        cv2.imshow("ROI debug", roi_dbg)

        if out is not None:
            out.write(output)

        #loop incrementing frames until time is greater than 1 second to get FPS
        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            print(f"Live FPS: {frames/(now-t0):.2f}")
            #reset frames and time zero
            t0 = now
            frames = 0

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    cap.release()
    if out is not None:
        out.release()
        print(f"Saved: {OUTPUT_PATH}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()