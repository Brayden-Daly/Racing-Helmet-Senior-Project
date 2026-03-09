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
OUTPUT_PATH = Path.home() / "local_work_tegra" / "track_polyfit_output.mp4"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = str(OUTPUT_PATH)

# ROI trapezoid (tune as needed)
ROI = dict(
    keep_trapezoid=True,
    cut_car=True,

    top_y=0.17,
    bottom_y=0.85,     # was 0.40 (too shallow)

    left_bottom=0.00,
    right_bottom=1.00,
    left_top=0.25,
    right_top=0.75,

    # aggressive car cut:
    car_cut_y=0.50,
    car_left=0.10,
    car_right=0.90,
)

# Painted-line mask (HSV)
# (Use these everywhere; tune in one place.)
WHITE_LOWER = (0, 0, 130)
WHITE_UPPER = (179, 70, 255)

# If your track has yellow lines, enable this
USE_YELLOW = False
YELLOW_LOWER = (15, 80, 120)
YELLOW_UPPER = (40, 255, 255)

# Mask cleanup
DO_CLOSE = True
CLOSE_K = 7
DO_OPEN = False
OPEN_K = 3

# Wall/sign suppression
WALL_CUT_FRAC = 0.88   # zero out mask to the right of this fraction of width (tune 0.85..0.92)
SKIP_TOP_BAND = 0.08   # skip top of ROI in fitting (tune 0.06..0.12)

# Polynomial fitting
POLY_DEGREE = 2
MIN_POINTS_PER_SIDE = 60
ROW_STEP = 2

# Temporal smoothing (EMA on polynomial coefficients)
USE_TEMPORAL_SMOOTH = True
EMA_ALPHA = 0.25

# Draw
CURVE_THICKNESS = 7
CURVE_COLOR = (0, 0, 255)

# Guidance cue
Y_TARGET_FRAC = 0.48
GUIDE_COLOR = (0, 255, 255)

# Debug windows
SHOW_DEBUG = True


# ----------------------------
# Helpers
# ----------------------------
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


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
            (int(right_bottom * w), int(bottom_y * h)),
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

    if img.ndim == 2:
        return cv2.bitwise_and(img, img, mask=mask)
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

    if roi.get("cut_car", False):
        car_poly = np.array([[
            (int(roi["car_left"] * w),  int(1.00 * h)),
            (int(roi["car_left"] * w),  int(roi["car_cut_y"] * h)),
            (int(roi["car_right"] * w), int(roi["car_cut_y"] * h)),
            (int(roi["car_right"] * w), int(1.00 * h)),
        ]], dtype=np.int32)
        cv2.polylines(out, car_poly, isClosed=True, color=(255, 255, 0), thickness=2)

    return out


def morph_close(bw_u8, k=5):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bw_u8, cv2.MORPH_CLOSE, kernel)


def morph_open(bw_u8, k=3):
    kernel = np.ones((k, k), np.uint8)
    return cv2.morphologyEx(bw_u8, cv2.MORPH_OPEN, kernel)


def fit_poly_x_of_y(ys, xs, degree=2):
    if ys.size < degree + 1:
        return None
    return np.polyfit(ys, xs, degree)


def eval_poly(coeffs, y):
    return float(np.polyval(coeffs, y))


def draw_poly_curve(img, coeffs, y0, y1, step=2, color=(0, 0, 255), thickness=6):
    h, w = img.shape[:2]
    pts = []
    for y in range(int(y0), int(y1) + 1, step):
        x = int(round(eval_poly(coeffs, y)))
        if 0 <= x < w:
            pts.append((x, y))
    if len(pts) >= 2:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], isClosed=False, color=color, thickness=thickness)


class PolyTracker:
    def __init__(self):
        self.left = None
        self.right = None

    def update(self, left_coeffs, right_coeffs, alpha=0.25):
        if left_coeffs is not None:
            self.left = left_coeffs if self.left is None else (alpha * left_coeffs + (1 - alpha) * self.left)
        if right_coeffs is not None:
            self.right = right_coeffs if self.right is None else (alpha * right_coeffs + (1 - alpha) * self.right)
        return self.left, self.right


tracker = PolyTracker()


# ----------------------------
# Pipeline
# ----------------------------
def process_frame(frame_bgr):
    h, w = frame_bgr.shape[:2]

    roi_frame = apply_roi_mask(frame_bgr, **ROI)
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)

    # Paint mask using config thresholds
    bw = cv2.inRange(hsv, WHITE_LOWER, WHITE_UPPER)
    if USE_YELLOW:
        yellow = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
        bw = cv2.bitwise_or(bw, yellow)

    # Kill far-right wall/sign pixels in the mask (big stability boost)
    bw[:, int(WALL_CUT_FRAC * w):] = 0

    # Morph cleanup
    if DO_CLOSE:
        bw = morph_close(bw, CLOSE_K)
    if DO_OPEN:
        bw = morph_open(bw, OPEN_K)

    # Fit region (skip top band where boards dominate)
    y_top = int(np.clip((ROI["top_y"] + SKIP_TOP_BAND) * h, 0, h - 1))
    y_bot = int(np.clip(ROI["bottom_y"] * h, 0, h - 1))
    if y_bot <= y_top:
        y_top, y_bot = 0, h - 1

    mid = w // 2
    right_x_max = int(WALL_CUT_FRAC * w)  # match the hard cut

    def collect_points_row_sampling_inner(bw_u8, y_min, y_max, x_min, x_max,
                                          row_step=2, trim_band=(10, 90), side="left"):
        """
        Pick the INNER boundary (closest to center):
          - left side: choose rightmost in left half (high percentile)
          - right side: choose leftmost in right half (low percentile)
        """
        ys, xs = [], []
        y0 = max(0, int(y_min))
        y1 = min(h - 1, int(y_max))
        lo_trim, hi_trim = trim_band

        for y in range(y0, y1 + 1, row_step):
            x_idx = np.flatnonzero(bw_u8[y, :])
            if x_idx.size == 0:
                continue

            x_idx = x_idx[(x_idx >= x_min) & (x_idx <= x_max)]
            if x_idx.size < 4:
                continue

            lo = np.percentile(x_idx, lo_trim)
            hi = np.percentile(x_idx, hi_trim)
            x_idx = x_idx[(x_idx >= lo) & (x_idx <= hi)]
            if x_idx.size < 4:
                continue

            if side == "left":
                x_pick = int(np.percentile(x_idx, 90))
            else:
                x_pick = int(np.percentile(x_idx, 10))

            ys.append(y)
            xs.append(x_pick)

        return np.array(ys, np.float32), np.array(xs, np.float32)

    # Collect points
    ly, lx = collect_points_row_sampling_inner(bw, y_top, y_bot, 0, mid - 5, row_step=ROW_STEP, side="left")
    ry, rx = collect_points_row_sampling_inner(bw, y_top, y_bot, mid + 5, right_x_max, row_step=ROW_STEP, side="right")

    left_coeffs = fit_poly_x_of_y(ly, lx, degree=POLY_DEGREE) if lx.size >= MIN_POINTS_PER_SIDE else None
    right_coeffs = fit_poly_x_of_y(ry, rx, degree=POLY_DEGREE) if rx.size >= MIN_POINTS_PER_SIDE else None

    if USE_TEMPORAL_SMOOTH:
        left_coeffs, right_coeffs = tracker.update(left_coeffs, right_coeffs, alpha=EMA_ALPHA)

    output = frame_bgr.copy()

    # Draw curves
    if left_coeffs is not None:
        draw_poly_curve(output, left_coeffs, y_top, y_bot, step=2, color=CURVE_COLOR, thickness=CURVE_THICKNESS)
    if right_coeffs is not None:
        draw_poly_curve(output, right_coeffs, y_top, y_bot, step=2, color=CURVE_COLOR, thickness=CURVE_THICKNESS)

    # Guidance cue (lookahead blend reduces jitter)
    y_target = int(np.clip(int(Y_TARGET_FRAC * h), y_top, y_bot))
    if left_coeffs is not None and right_coeffs is not None:
        y2 = int(np.clip(y_target - 40, y_top, y_bot))

        x_mid_1 = 0.5 * (eval_poly(left_coeffs, y_target) + eval_poly(right_coeffs, y_target))
        x_mid_2 = 0.5 * (eval_poly(left_coeffs, y2)       + eval_poly(right_coeffs, y2))
        x_mid = int(np.clip(0.6 * x_mid_2 + 0.4 * x_mid_1, 0, w - 1))

        cv2.circle(output, (x_mid, y_target), 9, GUIDE_COLOR, -1)
        cv2.arrowedLine(output, (w // 2, y_bot), (x_mid, y_target),
                        GUIDE_COLOR, 3, tipLength=0.25)

    # Debug text
    cv2.putText(output,
                f"Lpts={lx.size} fit={'Y' if left_coeffs is not None else 'N'} | "
                f"Rpts={rx.size} fit={'Y' if right_coeffs is not None else 'N'}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return output, bw


# ----------------------------
# Main loop
# ----------------------------
def main():
    cap = cv2.VideoCapture(PATH_OF_VID)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {PATH_OF_VID}")

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

        frame = resize_frame(frame, FRAME_WIDTH, FRAME_HEIGHT)

        output, mask_dbg = process_frame(frame)
        roi_dbg = draw_roi_outline(frame, ROI)

        cv2.imshow("Apex overlay (q/ESC to quit)", output)
        if SHOW_DEBUG:
            cv2.imshow("Paint mask (ROI)", mask_dbg)
            cv2.imshow("ROI debug", roi_dbg)

        if out is not None:
            out.write(output)

        frames += 1
        now = time.time()
        if now - t0 >= 1.0:
            print(f"Live FPS: {frames/(now-t0):.2f}")
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