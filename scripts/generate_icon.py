"""Generate a clean microphone tray icon for Wispr Local."""
from PIL import Image, ImageDraw
import os

def draw_mic_icon(size, padding_ratio=0.15):
    """Draw a minimal microphone icon on transparent background with purple accent."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    pad = int(size * padding_ratio)
    w = size - 2 * pad
    h = size - 2 * pad
    cx = size // 2

    # Colors
    purple = (168, 85, 247, 255)       # #a855f7
    light_purple = (196, 140, 255, 255)

    # Line width scales with size
    lw = max(1, int(size * 0.06))

    # --- Microphone body (rounded rectangle / capsule) ---
    mic_w = int(w * 0.30)
    mic_h = int(h * 0.45)
    mic_left = cx - mic_w // 2
    mic_top = pad
    mic_right = cx + mic_w // 2
    mic_bottom = mic_top + mic_h

    # Draw capsule shape (rectangle + semicircle top + flat bottom)
    radius = mic_w // 2
    # Top semicircle
    draw.ellipse(
        [mic_left, mic_top, mic_right, mic_top + mic_w],
        fill=purple,
    )
    # Body rectangle
    draw.rectangle(
        [mic_left, mic_top + radius, mic_right, mic_bottom],
        fill=purple,
    )
    # Bottom semicircle
    draw.ellipse(
        [mic_left, mic_bottom - radius, mic_right, mic_bottom + radius],
        fill=purple,
    )

    # --- Arc (U-shape around mic) ---
    arc_margin = int(w * 0.05)
    arc_left = mic_left - int(w * 0.12) - arc_margin
    arc_right = mic_right + int(w * 0.12) + arc_margin
    arc_top = mic_top + int(mic_h * 0.25)
    arc_bottom_y = mic_bottom + int(h * 0.12)
    arc_h = arc_bottom_y - arc_top

    draw.arc(
        [arc_left, arc_top, arc_right, arc_top + arc_h * 2],
        start=0,
        end=180,
        fill=light_purple,
        width=lw,
    )

    # --- Vertical stem ---
    stem_top = arc_top + arc_h
    stem_bottom = stem_top + int(h * 0.15)
    draw.line(
        [(cx, stem_top), (cx, stem_bottom)],
        fill=light_purple,
        width=lw,
    )

    # --- Horizontal base ---
    base_w = int(w * 0.25)
    draw.line(
        [(cx - base_w // 2, stem_bottom), (cx + base_w // 2, stem_bottom)],
        fill=light_purple,
        width=lw,
    )

    return img


def main():
    icons_dir = os.path.join(os.path.dirname(__file__), "src-tauri", "icons")
    os.makedirs(icons_dir, exist_ok=True)

    # Generate multiple sizes
    sizes = [16, 32, 48, 64, 128, 256]
    images = {}
    for s in sizes:
        images[s] = draw_mic_icon(s)

    # Save 32x32 PNG (for tray)
    images[32].save(os.path.join(icons_dir, "32x32.png"))
    print("Saved 32x32.png")

    # Save main icon.png (256x256)
    images[256].save(os.path.join(icons_dir, "icon.png"))
    print("Saved icon.png")

    # Save ICO with multiple sizes
    ico_path = os.path.join(icons_dir, "icon.ico")
    # ICO: save the 256 as base, include all sizes
    images[256].save(
        ico_path,
        format="ICO",
        sizes=[(s, s) for s in sizes],
    )
    print(f"Saved icon.ico with sizes: {sizes}")


if __name__ == "__main__":
    main()
