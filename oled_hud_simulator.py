#!/usr/bin/env python3
"""
OLED-only HUD simulator.

This does NOT run GPS, video, edge detection, or any OpenCV windows.
It only shows the same OLED HUD screens used by the main file:
  1. LEFT turn with countdown
  2. TRACK CLEAR
  3. RIGHT turn with countdown
  4. TRACK CLEAR
Then it loops forever.
"""

import time
import logging
from PIL import Image, ImageDraw, ImageFont

from waveshare_OLED import OLED_1in51

logging.basicConfig(level=logging.INFO)

# Match the OLED HUD settings from the main file.
OLED_ROTATE_180 = True
OLED_DOT_BLINK_S = 0.5
START_DIST_M = 100.0

# Simulation timing.
COUNTDOWN_START_M = 100
COUNTDOWN_STEP_M = 5
COUNTDOWN_FRAME_S = 0.20
CLEAR_SCREEN_S = 2.0


def draw_oled_right_arrow(draw, cx, cy):
    """Same thick right arrow used by the main OLED HUD."""
    draw.rectangle((cx - 40, cy - 6, cx + 15, cy + 6), fill=0)
    draw.polygon([
        (cx + 15, cy - 22),
        (cx + 45, cy),
        (cx + 15, cy + 22),
    ], fill=0)


def draw_oled_left_arrow(draw, cx, cy):
    """Same thick left arrow used by the main OLED HUD."""
    draw.rectangle((cx - 15, cy - 6, cx + 40, cy + 6), fill=0)
    draw.polygon([
        (cx - 15, cy - 22),
        (cx - 45, cy),
        (cx - 15, cy + 22),
    ], fill=0)


def draw_oled_blink_dot(draw, width, dot_on):
    """Same blinking top-right dot used by the main OLED HUD."""
    if dot_on:
        draw.ellipse((width - 18, 4, width - 4, 18), fill=0)


class OLEDHUDSimulator:
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

    def _update_blink_dot(self):
        now = time.time()
        if now - self.last_dot_switch >= OLED_DOT_BLINK_S:
            self.dot_on = not self.dot_on
            self.last_dot_switch = now

    def _text_center(self, draw, text, y, font=None, fill=0):
        font = font or self.font_small
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
        except Exception:
            tw = len(text) * 6
        draw.text(((self.width - tw) // 2, y), text, font=font, fill=fill)

    def show_turn(self, direction, distance_m):
        """Show exactly the main OLED HUD turn layout: arrow, TURN text, distance, dot."""
        self._update_blink_dot()

        direction = direction.upper()
        image = Image.new("1", (self.width, self.height), "WHITE")
        draw = ImageDraw.Draw(image)

        if direction == "LEFT":
            draw_oled_left_arrow(draw, self.cx, self.cy - 10)
        else:
            draw_oled_right_arrow(draw, self.cx, self.cy - 10)

        self._text_center(draw, f"TURN {direction}", 2)
        self._text_center(draw, f"{max(distance_m, 0):.0f} m", self.height - 18)
        draw_oled_blink_dot(draw, self.width, self.dot_on)

        if OLED_ROTATE_180:
            image = image.rotate(180)

        self.disp.ShowImage(self.disp.getbuffer(image))

    def show_clear(self):
        """Show exactly the main OLED HUD idle layout: TRACK CLEAR, GPS/status line, dot."""
        self._update_blink_dot()

        image = Image.new("1", (self.width, self.height), "WHITE")
        draw = ImageDraw.Draw(image)

        self._text_center(draw, "TRACK CLEAR", self.cy - 8)
        self._text_center(draw, "Sim HUD only", self.height - 18)
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


def run_countdown(hud, direction):
    for dist_m in range(COUNTDOWN_START_M, -1, -COUNTDOWN_STEP_M):
        hud.show_turn(direction, dist_m)
        time.sleep(COUNTDOWN_FRAME_S)


def run_clear(hud):
    end_time = time.time() + CLEAR_SCREEN_S
    while time.time() < end_time:
        hud.show_clear()
        time.sleep(0.05)


def main():
    hud = OLEDHUDSimulator()
    logging.info("OLED HUD simulator running. Press Ctrl+C to stop.")

    try:
        while True:
            run_countdown(hud, "LEFT")
            run_clear(hud)
            run_countdown(hud, "RIGHT")
            run_clear(hud)
    except KeyboardInterrupt:
        logging.info("ctrl + c")
    finally:
        hud.close()


if __name__ == "__main__":
    main()
