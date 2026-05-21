#!/usr/bin/python3

import time
import logging
from PIL import Image, ImageDraw

from waveshare_OLED import OLED_1in51

logging.basicConfig(level=logging.INFO)


def draw_right_arrow(draw, cx, cy):
    """
    Draw a thick arrow pointing right.
    cx, cy = center of screen
    """
    # Main shaft
    draw.rectangle((cx - 40, cy - 6, cx + 15, cy + 6), fill=0)

    # Arrow head
    draw.polygon([
        (cx + 15, cy - 22),
        (cx + 45, cy),
        (cx + 15, cy + 22)
    ], fill=0)


def draw_left_arrow(draw, cx, cy):
    """
    Draw a thick arrow pointing left.
    cx, cy = center of screen
    """
    # Main shaft
    draw.rectangle((cx - 15, cy - 6, cx + 40, cy + 6), fill=0)

    # Arrow head
    draw.polygon([
        (cx - 15, cy - 22),
        (cx - 45, cy),
        (cx - 15, cy + 22)
    ], fill=0)


def draw_blink_dot(draw, width, dot_on):
    """
    Draw blinking dot in top-right corner.
    """
    if dot_on:
        # Large filled circle
        draw.ellipse((width - 18, 4, width - 4, 18), fill=0)


try:
    disp = OLED_1in51.OLED_1in51()

    logging.info("1.51inch OLED")
    disp.Init()
    disp.clear()

    width = disp.width
    height = disp.height

    cx = width // 2
    cy = height // 2

    arrow_direction = "right"
    dot_on = True

    last_arrow_switch = time.time()
    last_dot_switch = time.time()

    while True:
        current_time = time.time()

        # Switch arrow direction every 2 seconds
        if current_time - last_arrow_switch >= 2.0:
            if arrow_direction == "right":
                arrow_direction = "left"
            else:
                arrow_direction = "right"

            last_arrow_switch = current_time

        # Blink dot every 0.5 seconds
        if current_time - last_dot_switch >= 0.5:
            dot_on = not dot_on
            last_dot_switch = current_time

        # Create blank white screen
        image = Image.new("1", (width, height), "WHITE")
        draw = ImageDraw.Draw(image)

        # Draw arrow in center
        if arrow_direction == "right":
            draw_right_arrow(draw, cx, cy)
        else:
            draw_left_arrow(draw, cx, cy)

        # Draw blinking dot
        draw_blink_dot(draw, width, dot_on)

        # Rotate if your display is upside down
        image = image.rotate(180)

        # Show image
        disp.ShowImage(disp.getbuffer(image))

        # Small delay to avoid maxing out CPU
        time.sleep(0.05)

except IOError as e:
    logging.info(e)

except KeyboardInterrupt:
    logging.info("ctrl + c")
    disp.clear()
    disp.module_exit()
    exit()
