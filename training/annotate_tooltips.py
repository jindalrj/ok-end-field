"""
Auto-annotate real tooltip training data using Tesseract + known item name matching.

For each tooltip crop, runs current Tesseract OCR on the title band,
then fuzzy-matches against known item names to produce ground truth.

Items that can't be confidently matched are left as __NEEDS_ANNOTATION__
for manual review.

Usage:
    python training/annotate_tooltips.py
"""

import sys
from pathlib import Path
from difflib import SequenceMatcher

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data.zh_en import ITEM_GAME_ENGLISH

GT_DIR = PROJECT_ROOT / "training" / "tesstrain" / "ground-truth"
ALL_NAMES = sorted(set(ITEM_GAME_ENGLISH.values()))
# Also include categories and common tooltip text
KNOWN_TEXTS = ALL_NAMES + [
    "Material", "Consumable", "Component", "Battery", "Powder",
    "Filled with Clean Water", "Filled with Resilient Water",
    "Filled with Shangshu Water", "Filled with Warm Spring Water",
    "Filled with Crimson Water", "Filled with Originium Water",
]


def best_match(ocr_text: str) -> tuple[str, float]:
    """Find best matching known text for OCR output."""
    if not ocr_text:
        return "", 0.0
    ocr_lower = ocr_text.lower().strip()
    best = ""
    best_score = 0.0
    for name in KNOWN_TEXTS:
        score = SequenceMatcher(None, ocr_lower, name.lower()).ratio()
        if score > best_score:
            best_score = score
            best = name
    return best, best_score


def main():
    # Try to init tesseract for OCR
    try:
        from ocr.tesseract_ocr import ensure_tesseract, ocr_text
        ensure_tesseract()
        has_tess = True
    except Exception as e:
        print(f"Tesseract not available ({e}), using filename-based annotation only")
        has_tess = False
        ocr_text = None

    gt_files = sorted(GT_DIR.glob("endfield_game_real_*.gt.txt"))
    if not gt_files:
        print(f"No real tooltip GT files in {GT_DIR}")
        print("Run generate_training_data.py first")
        return 1

    annotated = 0
    skipped = 0
    already_done = 0

    for gt_file in gt_files:
        current = gt_file.read_text(encoding="utf-8").strip()
        if current != "__NEEDS_ANNOTATION__":
            already_done += 1
            continue

        tif_file = gt_file.with_suffix("").with_suffix(".tif")
        if not tif_file.exists():
            skipped += 1
            continue

        # Run OCR on the tif
        detected = ""
        if has_tess:
            img = cv2.imread(str(tif_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Invert if needed (tesseract expects dark text on light bg)
                if img.mean() < 128:
                    img = 255 - img
                import pytesseract
                detected = pytesseract.image_to_string(img, config="--psm 7").strip()

        if detected:
            match, score = best_match(detected)
            if score >= 0.75:
                gt_file.write_text(match, encoding="utf-8")
                annotated += 1
            else:
                # Low confidence - leave for manual review
                gt_file.write_text(f"__REVIEW__:{detected}", encoding="utf-8")
                skipped += 1
        else:
            skipped += 1

    print(f"Results:")
    print(f"  Auto-annotated: {annotated}")
    print(f"  Need manual review: {skipped}")
    print(f"  Already done: {already_done}")
    print(f"  Total: {len(gt_files)}")

    if skipped > 0:
        print(f"\nManually review files marked __REVIEW__ or __NEEDS_ANNOTATION__ in:")
        print(f"  {GT_DIR}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
