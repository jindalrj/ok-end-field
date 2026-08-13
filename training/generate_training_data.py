"""
Generate Tesseract training data from:
1. Real tooltip crops (title band) with ground truth from OCR + manual verification
2. Synthetic renders of all known item names matching the game's font style

Output: training/tesstrain/ground-truth/ with .tif + .gt.txt pairs
       ready for tesstrain Makefile.

Usage:
    python training/generate_training_data.py

Prerequisites:
    pip install pillow opencv-python numpy
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.zh_en import ITEM_GAME_ENGLISH

OUTPUT_DIR = PROJECT_ROOT / "training" / "tesstrain" / "ground-truth"
TOOLTIP_DIR = Path(os.environ.get("TOOLTIP_DIR", PROJECT_ROOT.parent / "ocr"))

# All known English item names (these are the OCR targets)
ALL_ITEM_NAMES = sorted(set(ITEM_GAME_ENGLISH.values()))

# Category names that appear in tooltip line 2
CATEGORIES = ["Material", "Consumable", "Component", "Battery", "Powder"]

# Description prefixes
DESCRIPTIONS = [
    "Filled with Clean Water",
    "Filled with Resilient Water",
    "Filled with Shangshu Water",
    "Filled with Warm Spring Water",
    "Filled with Crimson Water",
    "Filled with Originium Water",
    "Filled with Fluorescent Water",
    "Filled with Karst Water",
    "Filled with High-Crystal Solution",
    "Filled with Carbon Solution",
    "Filled with Stable Carbon Solution",
]


def crop_title_band(tooltip_path: Path) -> np.ndarray | None:
    """Extract the title text band (top 33%) from a tooltip crop."""
    img = cv2.imread(str(tooltip_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    band_h = int(h * 0.40)  # top 40% to get full title
    title_crop = img[0:band_h, 0:w]
    # Verify it has content (not all black)
    if title_crop.mean() < 10:
        return None
    return title_crop


def render_synthetic(text: str, width: int = 650, height: int = 70,
                     font_size: int = 38) -> np.ndarray:
    """Render text matching game tooltip style: white text on dark grey bg."""
    img = Image.new("RGB", (width, height), color=(45, 45, 50))
    draw = ImageDraw.Draw(img)

    # Try to use a sans-serif font similar to the game
    font = None
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Draw text with slight padding from left
    draw.text((20, (height - font_size) // 2), text, fill=(255, 255, 255), font=font)

    return np.array(img)


def save_training_pair(img: np.ndarray, ground_truth: str, name: str):
    """Save a .tif + .gt.txt pair for tesstrain."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # tesstrain expects: <lang>_<fontname>_<id>.tif + .gt.txt
    base = f"endfield_game_{name}"
    tif_path = OUTPUT_DIR / f"{base}.tif"
    gt_path = OUTPUT_DIR / f"{base}.gt.txt"

    # Convert to grayscale, resize to reasonable DPI for training
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Binarize with OTSU for cleaner training
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    cv2.imwrite(str(tif_path), binary)
    gt_path.write_text(ground_truth, encoding="utf-8")


def generate_from_tooltips():
    """Generate training data from real tooltip crops.

    Uses the tooltip filename pattern to determine ground truth:
    r{round}_tooltip_{row}_{col}.png

    For now, we generate title-band crops and will need manual GT annotation
    or use the known item list to match.
    """
    tooltip_files = sorted(TOOLTIP_DIR.glob("r*_tooltip_*.png"))
    if not tooltip_files:
        print(f"No tooltip files found in {TOOLTIP_DIR}")
        return 0

    count = 0
    for tf in tooltip_files:
        title_crop = crop_title_band(tf)
        if title_crop is None:
            continue
        # Save with placeholder GT - these need manual annotation
        # File name: r0_tooltip_0_0.png -> real_r0_0_0
        stem = tf.stem.replace("tooltip_", "")
        save_training_pair(title_crop, "__NEEDS_ANNOTATION__", f"real_{stem}")
        count += 1

    return count


def generate_synthetic():
    """Generate synthetic training images for all known item names."""
    count = 0

    # Item names
    for i, name in enumerate(ALL_ITEM_NAMES):
        img = render_synthetic(name)
        save_training_pair(img, name, f"synth_item_{i:03d}")
        count += 1

    # Categories
    for i, cat in enumerate(CATEGORIES):
        img = render_synthetic(cat, font_size=28, height=50)
        save_training_pair(img, cat, f"synth_cat_{i:02d}")
        count += 1

    # Descriptions (for filled containers)
    for i, desc in enumerate(DESCRIPTIONS):
        img = render_synthetic(desc, width=700, font_size=28, height=50)
        save_training_pair(img, desc, f"synth_desc_{i:02d}")
        count += 1

    # Grade markers
    for grade in ["A", "B", "C"]:
        for prefix in ["", "[", "] "]:
            text = f"[{grade}]" if not prefix else f"{prefix}{grade}"
            img = render_synthetic(text, width=200, font_size=36, height=60)
            save_training_pair(img, f"[{grade}]", f"synth_grade_{grade}")
        count += 1

    # Battery prefixes with names
    for prefix in ["LC", "SC", "HC"]:
        for suffix in ["Valley Battery", "Wuling Battery"]:
            text = f"{prefix} {suffix}"
            img = render_synthetic(text)
            save_training_pair(img, text, f"synth_batt_{prefix}_{suffix.split()[0]}")
            count += 1

    # Count numbers (appear below items)
    for num in ["1", "2", "5", "10", "14", "23", "26", "31", "36", "77", "91",
                "150", "207", "449", "4", "27", "80K", "79.9K", "5.39K", "12K",
                "35K", "50.7K", "78.6K", "1.2M"]:
        img = render_synthetic(num, width=200, font_size=32, height=50)
        save_training_pair(img, num, f"synth_count_{num.replace('.','d')}")
        count += 1

    return count


def main():
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tooltip source: {TOOLTIP_DIR}")
    print(f"Known items: {len(ALL_ITEM_NAMES)}")
    print()

    n_real = generate_from_tooltips()
    print(f"Generated {n_real} real tooltip training pairs (need annotation)")

    n_synth = generate_synthetic()
    print(f"Generated {n_synth} synthetic training pairs")

    print(f"\nTotal: {n_real + n_synth} training pairs in {OUTPUT_DIR}")
    print()
    print("Next steps:")
    print("1. Annotate real tooltip pairs: edit .gt.txt files in ground-truth/")
    print("   (replace __NEEDS_ANNOTATION__ with actual text)")
    print("2. Run training: cd training && make training")
    print("   OR: python training/run_training.py")


if __name__ == "__main__":
    main()
