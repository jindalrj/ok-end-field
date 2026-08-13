"""
Tesseract OCR integration for Endfield game UI text detection.

PPOCRv5 fails to detect English text in this game's stylized font.
Tesseract with --psm 6 (uniform text block) works reliably.

This module:
1. Auto-downloads portable Tesseract on first use (Windows)
2. Provides a simple ocr() function that returns detected text with positions
"""

import os
import re
import sys
import shutil
import logging
import subprocess
import zipfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Find bundled Tesseract by walking up from this file's location.
# At runtime the structure is: .../data/apps/ok-ef/tesseract/tesseract.exe
# and this file is at:          .../data/apps/ok-ef/<repo_or_working>/src/ocr/tesseract_ocr.py
# We don't know exact depth, so check each ancestor for a sibling tesseract/ dir.
def _find_bundled_dirs() -> list[Path]:
    """Walk up from __file__ and collect any ancestor/tesseract/ dirs."""
    dirs = []
    p = Path(__file__).resolve().parent
    for _ in range(6):  # up to 6 levels above src/ocr/
        p = p.parent
        candidate = p / "tesseract"
        if candidate.is_dir():
            dirs.append(candidate)
    return dirs

_BUNDLED_DIRS = _find_bundled_dirs()

# Fallback cache directory for Tesseract (auto-download location)
_CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ok-ef" / "tesseract"
_TESSERACT_EXE = _CACHE_DIR / "tesseract.exe"

_initialized = False


def _find_tesseract() -> str | None:
    """Find tesseract.exe - check bundled, cache, PATH, and common locations."""
    # 1. Check bundled with app (placed by build workflow)
    for bundled_dir in _BUNDLED_DIRS:
        exe = bundled_dir / "tesseract.exe"
        if exe.exists():
            return str(exe)
        # Also check one level deeper (Copy-Item may nest the folder)
        if bundled_dir.exists():
            for sub_exe in bundled_dir.rglob("tesseract.exe"):
                return str(sub_exe)

    # 2. Check download cache
    if _TESSERACT_EXE.exists():
        return str(_TESSERACT_EXE)

    # 3. Check PATH
    tesseract_in_path = shutil.which("tesseract")
    if tesseract_in_path:
        return tesseract_in_path

    # 4. Check common Windows install locations
    common_paths = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    ]
    for p in common_paths:
        if p.exists():
            return str(p)

    return None


def _download_tesseract():
    """Download portable Tesseract for Windows using the UB-Mannheim installer in silent mode."""
    import urllib.request

    logger.info("Tesseract not found. Downloading and installing...")
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Use UB-Mannheim installer in silent mode to install to our cache dir
    installer_url = "https://github.com/UB-Mannheim/tesseract/releases/download/v5.3.4.20240503/tesseract-ocr-w64-setup-5.3.4.20240503.exe"
    installer_path = _CACHE_DIR / "tesseract-installer.exe"

    try:
        logger.info(f"Downloading Tesseract installer...")
        urllib.request.urlretrieve(installer_url, str(installer_path))

        # Run installer silently to our cache directory
        logger.info(f"Installing to {_CACHE_DIR}...")
        result = subprocess.run(
            [str(installer_path), "/S", f"/D={_CACHE_DIR}"],
            timeout=120,
            capture_output=True,
        )

        # Clean up installer
        installer_path.unlink(missing_ok=True)

        if _TESSERACT_EXE.exists():
            logger.info(f"Tesseract installed successfully at {_TESSERACT_EXE}")
        else:
            # Installer may place it in a subdirectory
            for exe in _CACHE_DIR.rglob("tesseract.exe"):
                logger.info(f"Found tesseract at {exe}")
                return

            raise FileNotFoundError("tesseract.exe not found after installation")

    except subprocess.TimeoutExpired:
        installer_path.unlink(missing_ok=True)
        raise RuntimeError("Tesseract installer timed out")
    except Exception as e:
        installer_path.unlink(missing_ok=True)
        logger.error(f"Failed to install Tesseract: {e}")
        raise RuntimeError(
            "Could not install Tesseract OCR automatically. "
            "Please install manually from https://github.com/UB-Mannheim/tesseract/wiki "
            "and ensure tesseract.exe is in PATH."
        ) from e


def ensure_tesseract() -> str:
    """Ensure Tesseract is available, downloading if needed.

    Returns a diagnostic string describing what was found/configured.
    Raises RuntimeError if tesseract cannot be found or installed.
    """
    global _initialized
    if _initialized:
        import pytesseract
        return f"already initialized, cmd={pytesseract.pytesseract.tesseract_cmd}"

    tesseract_path = _find_tesseract()
    diag_parts = [f"search={'found' if tesseract_path else 'NOT_FOUND'}"]
    # Log where we looked so path issues are obvious
    diag_parts.append(f"__file__={__file__}")
    diag_parts.append(f"bundled={[str(d) for d in _BUNDLED_DIRS]}")
    if tesseract_path:
        diag_parts.append(f"path={tesseract_path}")
    else:
        if sys.platform == "win32":
            _download_tesseract()
            tesseract_path = str(_TESSERACT_EXE)
            diag_parts.append(f"downloaded_to={tesseract_path}")
        else:
            raise RuntimeError(
                "Tesseract not found. Install via: brew install tesseract (macOS) "
                "or apt install tesseract-ocr (Linux)"
            )

    # Configure pytesseract - resolve to absolute native path (avoids mixed slashes on Windows)
    import pytesseract
    tesseract_path = str(Path(tesseract_path).resolve())
    pytesseract.pytesseract.tesseract_cmd = tesseract_path

    # Set TESSDATA_PREFIX to the tessdata directory itself (not its parent).
    # Must use OS-native path separators on Windows to avoid mixed-slash errors.
    tessdata = Path(tesseract_path).resolve().parent / "tessdata"
    if tessdata.exists():
        os.environ["TESSDATA_PREFIX"] = str(tessdata)
        diag_parts.append(f"tessdata=OK({tessdata})")
    else:
        diag_parts.append(f"tessdata=MISSING({tessdata})")

    # Verify tesseract actually runs
    try:
        result = subprocess.run(
            [tesseract_path, "--version"],
            capture_output=True, text=True, timeout=5
        )
        ver = result.stdout.splitlines()[0] if result.stdout else "unknown"
        diag_parts.append(f"version={ver}")
    except Exception as e:
        diag_parts.append(f"version_check_FAILED={e}")

    _initialized = True
    return ", ".join(diag_parts)


def ocr_frame(frame: np.ndarray, box=None, psm: int = 6) -> list[dict]:
    """
    Run Tesseract OCR on a frame (or cropped region).

    Args:
        frame: BGR or RGB numpy array (full game frame)
        box: Optional (x1, y1, x2, y2) pixel coordinates to crop
        psm: Page segmentation mode (6=uniform block, 11=sparse text)

    Returns:
        List of dicts: [{"text": str, "confidence": int, "x": int, "y": int, "w": int, "h": int}]
    """
    if not _initialized:
        return []

    try:
        import pytesseract
        from PIL import Image

        if frame is None:
            return []

        # Convert BGR to RGB if needed (OpenCV uses BGR)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame[:, :, ::-1])  # BGR -> RGB
        else:
            img = Image.fromarray(frame)

        # Crop if box specified
        if box is not None:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            img = img.crop((x1, y1, x2, y2))
        else:
            x1, y1 = 0, 0

        img_np = np.array(img)
        if img_np.size == 0:
            return []

        data = pytesseract.image_to_data(img_np, output_type=pytesseract.Output.DICT, config=f'--psm {psm}')

        results = []
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = int(data['conf'][i])
            if text and conf > 30:
                results.append({
                    "text": text,
                    "confidence": conf,
                    "x": data['left'][i] + x1,
                    "y": data['top'][i] + y1,
                    "w": data['width'][i],
                    "h": data['height'][i],
                })
        return results
    except Exception as e:
        logger.debug(f"Tesseract ocr_frame failed: {e}")
        return []


# Diagnostic state accessible from the task via: from src.ocr.tesseract_ocr import _ocr_diag
_ocr_diag = {"calls": 0, "errors": [], "last_error": ""}


def ocr_text(frame: np.ndarray, box=None, psm: int = 6) -> str:
    """
    Run Tesseract OCR and return combined text string.

    Args:
        frame: BGR or RGB numpy array
        box: Optional (x1, y1, x2, y2) pixel coordinates to crop
        psm: Page segmentation mode

    Returns:
        Combined text string from all detections, or "" on any error
    """
    if not _initialized:
        _ocr_diag["last_error"] = "not_initialized"
        return ""

    try:
        import pytesseract
        from PIL import Image

        if frame is None:
            return ""

        if len(frame.shape) == 3 and frame.shape[2] == 3:
            img = Image.fromarray(frame[:, :, ::-1])
        else:
            img = Image.fromarray(frame)

        if box is not None:
            x1, y1, x2, y2 = box
            h, w = frame.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            img = img.crop((x1, y1, x2, y2))

        img_np = np.array(img)
        if img_np.size == 0:
            return ""

        _ocr_diag["calls"] += 1
        result = pytesseract.image_to_string(img_np, config=f'--psm {psm}').strip()
        return result
    except Exception as e:
        err_msg = f"{type(e).__name__}: {e}"
        _ocr_diag["last_error"] = err_msg
        if len(_ocr_diag["errors"]) < 5:
            _ocr_diag["errors"].append(err_msg)
        return ""


def ocr_match(frame: np.ndarray, box, target, psm: int = 6) -> bool:
    """
    Check if target text appears in the specified region.

    Args:
        frame: BGR numpy array (game frame)
        box: (x1, y1, x2, y2) pixel coordinates
        target: Text to search for — accepts str (substring match) or
                re.Pattern (regex match), both case-insensitive.

    Returns:
        True if target text found in region, False on any error
    """
    text = ocr_text(frame, box=box, psm=psm)
    if not text:
        return False
    if isinstance(target, re.Pattern):
        return bool(target.search(text))
    return target.lower() in text.lower()
