import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import win32api
import win32con
import win32gui
from qfluentwidgets import FluentIcon

from ok.feature.Box import Box
from src.data.world_map import item_to_warehouse_dict
from src.data.zh_en import ITEM_WAREHOUSE_CATEGORY_EN_BY_ZH, ITEM_TRANSLATION_DICT, ITEM_GAME_ENGLISH
from src.core.BaseEfTask import BaseEfTask
from src.icons import Icons
from src.ocr import ocr_text, ocr_match, ensure_tesseract
from src.ocr import tesseract_ocr as _tess_mod
from src.ocr.tesseract_ocr import _ocr_diag
from src.tasks.onetime.item_matcher import match_score as item_match_score, build_vocab

# Vocabulary of all words in known English item names - lets the matcher
# tell real leftover words ('ore', 'component') from OCR junk ('joo').
_MATCH_VOCAB = build_vocab(ITEM_GAME_ENGLISH.values())

# Maps Chinese item name -> template image filename (without .png) in assets/items/images/
_ITEM_TEMPLATE_DIR = Path(__file__).resolve().parents[3] / "assets" / "items" / "images"
_ITEM_JSON = Path(__file__).resolve().parents[3] / "assets" / "items" / "item.json"
_ITEM_TEMPLATE_MAP: dict[str, str] = {}

def _load_item_template_map() -> dict[str, str]:
    """Load Chinese name -> template filename mapping from item.json."""
    global _ITEM_TEMPLATE_MAP
    if _ITEM_TEMPLATE_MAP:
        return _ITEM_TEMPLATE_MAP
    try:
        data = json.loads(_ITEM_JSON.read_text(encoding="utf-8"))
        _ITEM_TEMPLATE_MAP = {v: k for k, v in data.items()}
    except Exception:
        _ITEM_TEMPLATE_MAP = {}
    return _ITEM_TEMPLATE_MAP

_LOCATIONS = {
    "zh_CN": {
        "valley4": "四号谷地",
        "wuling": "武陵",
    },
    "zh_TW": {
        "valley4": "四號谷地",
        "wuling": "武陵",
    },
    "en_US": {
        "valley4": "Valley IV",
        "wuling": "Wuling",
    },
}

# Detection patterns for current location (OCR text in the depot title)
_LOCATION_DETECT = {
    "zh_CN": {
        "wuling": "武陵仓库",
        "valley4": ("谷地", "仓库"),  # both substrings must appear
    },
    "zh_TW": {
        "wuling": "武陵倉庫",
        "valley4": ("谷地", "倉庫"),
    },
    "en_US": {
        "wuling": "Wuling Depot",
        "valley4": ("Valley", "Depot"),
    },
}


class WarehouseTransferTask(BaseEfTask):
    """
    背包物品跨仓库转移（发货仓库 -> 收货仓库 -> 一键存放 -> 切回发货仓库）。

    依赖：
    - OCR 用于识别：仓库标题/仓库切换按钮/确认/已连接/一键存放
    - template 用于识别：物品图标（来自 assets/items/images）
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "仓库物品转移"
        self.icon = Icons.ItemTransfer
        self.group_name = "仓储管理"
        self.group_icon = FluentIcon.FOLDER
        self.description = "从发货仓库取出指定物品，切到收货仓库后一键存放"

        self.default_config.update(
            {
                "发货仓库": "valley4",
                "收货仓库": "wuling",
                "物品": "Dense Originium Powder",
                "转移轮次": 10,
            }
        )
        self.config_description.update(
            {
                "发货仓库": "Source depot (transfer from)",
                "收货仓库": "Destination depot (transfer to)",
                "物品": "Item to transfer (English name)",
                "转移轮次": "Number of transfer rounds",
            }
        )
        _location_keys = list(_LOCATIONS.get("zh_CN", {}).keys())
        self.config_type["发货仓库"] = {"type": "drop_down", "options": _location_keys}
        self.config_type["收货仓库"] = {"type": "drop_down", "options": _location_keys}
        # Item dropdown uses English game names
        _en_items = sorted(ITEM_GAME_ENGLISH.values())
        self.config_type["物品"] = {
            "type": "drop_down",
            "options": _en_items,
        }
        self._template_cache: dict[str, object] = {}
        self._item_name_cache: dict[str, str] | None = None

    def _load_item_template(self, item_name: str):
        """Load the template image (BGR) and alpha mask for an item.

        Returns (template_bgr, mask) tuple or (None, None).
        The alpha channel from RGBA templates is used as the match mask,
        which is critical for correct template matching (without mask,
        confidence drops from 0.83+ to 0.53 due to transparent background).
        """
        if item_name in self._template_cache:
            return self._template_cache[item_name]
        template_map = _load_item_template_map()
        template_filename = template_map.get(item_name)
        if not template_filename:
            self._template_cache[item_name] = (None, None)
            return (None, None)
        template_path = _ITEM_TEMPLATE_DIR / f"{template_filename}.png"
        if not template_path.exists():
            self._template_cache[item_name] = (None, None)
            return (None, None)
        img = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            self._template_cache[item_name] = (None, None)
            return (None, None)
        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            bgr = img[:, :, :3]
            # Create 3-channel mask from alpha for cv2.matchTemplate
            mask = cv2.merge([alpha, alpha, alpha])
        else:
            bgr = img
            mask = None
        self._template_cache[item_name] = (bgr, mask)
        return (bgr, mask)

    def _to_one_type_page(self, item_name: str):
        # Resolve English game name to Chinese if needed for category lookup
        zh_name = item_name
        if item_name not in item_to_warehouse_dict:
            _en_to_zh = {v.lower(): k for k, v in ITEM_GAME_ENGLISH.items()}
            zh_name = _en_to_zh.get(item_name.lower(), item_name)
        category_zh = item_to_warehouse_dict.get(zh_name, "")
        category_en_name = ITEM_WAREHOUSE_CATEGORY_EN_BY_ZH.get(category_zh, "")
        if not category_en_name:
            self.log_info(f"[category] cannot find category for '{item_name}', skipping navigation")
            return
        result = self.find_feature(feature=f"{category_en_name}_icon")
        if not result:
            self.log_info(f"[category] icon '{category_en_name}_icon' not found, may already be on correct page")
        if result:
            self.click(result[0])
            self.wait_ui_stable(refresh_interval=0.2)

    def _get_locale(self) -> str:
        locale = self.runtime_locale
        if locale and locale in _LOCATIONS:
            return locale
        return "zh_CN"

    def _detect_current_location(self) -> str | None:
        """Detect current depot location using Tesseract OCR (primary) with framework OCR fallback."""
        self.next_frame()
        locale = self._get_locale()
        detect_map = _LOCATION_DETECT.get(locale, _LOCATION_DETECT["zh_CN"])
        frame = self.executor.frame

        debug_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ok-ef" / "grid_scan_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Primary: Tesseract OCR on the title region
        if frame is not None:
            h, w = frame.shape[:2]
            # Title area: "Valley IV Depot" / "武陵仓库". Measured on 4K frames:
            # title text rows y=0.190-0.217; decorative barcode glyphs sit at
            # y=0.173-0.182 and OCR as garbage ('479', '|', 'ill') if included,
            # so the top bound must be > 0.182. Icon box ends x<0.145.
            title_box = (int(w * 0.145), int(h * 0.187), int(w * 0.29), int(h * 0.219))
            text = ocr_text(frame, box=title_box, psm=7)

            # Debug: save annotated screenshot
            try:
                debug_frame = frame.copy()
                bx1, by1, bx2, by2 = title_box
                cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 3)
                label = f"Tess: '{text}'" if text else "Tess: (empty)"
                cv2.putText(debug_frame, label, (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                cv2.imwrite(str(debug_dir / "detect_location.png"), debug_frame)
                # Save cropped title region
                cv2.imwrite(str(debug_dir / "detect_location_crop.png"), frame[by1:by2, bx1:bx2])
            except Exception:
                pass

            if text:
                self.log_info(f"[detect_location] Tesseract: '{text}'")
                for loc_key, pattern in detect_map.items():
                    if isinstance(pattern, tuple):
                        if all(part.lower() in text.lower() for part in pattern):
                            return loc_key
                    elif pattern.lower() in text.lower():
                        return loc_key

        # Fallback: framework OCR (works for Chinese)
        detect_box = self.box_of_screen(0.02, 0.03, 0.29, 0.23, name="current_location_area")
        boxes = self.ocr(box=detect_box, threshold=0.1)
        all_texts = []
        for box in boxes or []:
            text = str(getattr(box, "name", "")).strip()
            if text:
                all_texts.append(text)
        combined_text = " ".join(all_texts)
        if all_texts:
            self.log_info(f"[detect_location] framework OCR: {all_texts}")

        for loc_key, pattern in detect_map.items():
            if isinstance(pattern, tuple):
                if all(part in combined_text for part in pattern):
                    return loc_key
            elif pattern in combined_text:
                return loc_key
        return None

    # Confirm/Connected button in the Switch Depot modal. Measured on 4K
    # frames: button pill spans (0.711-0.870, 0.779-0.836); its text
    # ("Confirm" / "Connected") sits at (0.775-0.850, 0.792-0.825).
    # A tight text-only box is required: including the dark surround makes
    # Otsu binarize pill-vs-background and the text is lost.
    _MODAL_BTN_TEXT_BOX = (0.775, 0.792, 0.850, 0.825)
    _MODAL_BTN_CLICK = (0.79, 0.81)

    @staticmethod
    def _debug_dir() -> Path:
        d = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ok-ef" / "grid_scan_debug"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _save_switch_debug(self, name: str, frame, boxes=None, dot=None):
        """Save an annotated debug screenshot.

        boxes: list of ((x1, y1, x2, y2), label) in pixel coords.
        dot: (x, y) pixel coords of the click that was performed.
        """
        try:
            debug_frame = frame.copy()
            for (bx1, by1, bx2, by2), label in (boxes or []):
                cv2.rectangle(debug_frame, (int(bx1), int(by1)), (int(bx2), int(by2)), (0, 255, 0), 3)
                if label:
                    cv2.putText(debug_frame, label, (int(bx1), int(by1) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            if dot is not None:
                cv2.circle(debug_frame, (int(dot[0]), int(dot[1])), 15, (0, 0, 255), -1)
            cv2.imwrite(str(self._debug_dir() / f"{name}.png"), debug_frame)
        except Exception:
            pass

    def _maybe_click_confirm(self, positional_fallback: bool = True):
        """Click Confirm button if it appears (Tesseract primary, positional fallback).

        positional_fallback=False: only click when OCR actually finds
        Confirm. Used after Quick Stash where usually NO confirm dialog
        appears - a blind click at (0.79, 0.81) lands in the backpack grid.
        """
        confirm_text = self.lang.WarehouseTransferTask.k_b56d9ac6
        self.next_frame()
        frame = self.executor.frame

        # Try Tesseract first
        if frame is not None:
            h, w = frame.shape[:2]
            l, t, r, b = self._MODAL_BTN_TEXT_BOX
            confirm_box = (int(w * l), int(h * t), int(w * r), int(h * b))
            text = ocr_text(frame, box=confirm_box, psm=7)
            click_px = (int(self._MODAL_BTN_CLICK[0] * w), int(self._MODAL_BTN_CLICK[1] * h))
            self._save_switch_debug("confirm_btn", frame,
                                    boxes=[(confirm_box, f"Tess: '{text}'")], dot=click_px)

            if ocr_match(frame, confirm_box, confirm_text):
                self.log_info(f"[confirm] Tesseract found '{text}', clicking")
                self.click_relative(*self._MODAL_BTN_CLICK)
                self.sleep(0.5)
                return True

        # Try framework OCR
        hits = self.wait_ocr(
            box=self.box_of_screen(0.70, 0.77, 0.88, 0.845, name="confirm_btn_area"),
            match=confirm_text,
            time_out=2,
            raise_if_not_found=False,
        )
        if hits:
            self.log_info(f"[confirm] framework OCR found, clicking")
            self.click(hits[0])
            self.sleep(0.5)
            return True

        if not positional_fallback:
            self.log_info("[confirm] OCR missed, no confirm present - skipping")
            return False

        # Positional fallback
        self.log_info("[confirm] OCR missed, clicking confirm position")
        self.click_relative(*self._MODAL_BTN_CLICK)
        self.sleep(0.5)
        return True

    def _find_depot_rows(self, frame) -> list[dict]:
        """Locate depot option rows in the Switch Depot modal.

        Rows are bright pills (val > 60) against the near-black modal
        background within (0.45-0.70, 0.35-0.68). Row positions SHIFT
        between modal states (measured: Wuling row at y 0.418 in the
        initial state vs 0.504 once selected), so rows are detected
        structurally instead of via fixed coordinates.

        The name text ("Wuling" / "Valley IV") occupies the top ~60% of
        each row, left of the Connected/Disconnected label. Its color
        varies (dark-on-white when selected, grey-on-grey when disabled,
        dark-on-teal when connected), so each name crop is Otsu-binarized
        and polarity-normalized before OCR.

        Returns list of {"text", "cx", "cy" (relative), "box" (pixel)}.
        """
        h, w = frame.shape[:2]
        x1, x2 = int(w * 0.45), int(w * 0.70)
        y1, y2 = int(h * 0.35), int(h * 0.68)
        hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        coverage = (val > 60).sum(axis=1) / (x2 - x1)
        rows = np.where(coverage > 0.5)[0]
        if len(rows) == 0:
            return []

        bands = []
        start = rows[0]
        for i in range(1, len(rows)):
            if rows[i] - rows[i - 1] > 10:
                bands.append((start, rows[i - 1]))
                start = rows[i]
        bands.append((start, rows[-1]))

        results = []
        for b_top, b_bot in bands:
            if b_bot - b_top < 40:
                continue
            ny1 = y1 + b_top + 2
            ny2 = y1 + b_top + int((b_bot - b_top) * 0.62)
            nx1, nx2 = int(w * 0.455), int(w * 0.66)
            crop = frame[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if (binary == 0).mean() > 0.5:
                binary = 255 - binary  # normalize to black text on white
            padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            text = ocr_text(padded, psm=7).strip()
            results.append({
                "text": text,
                "cx": 0.575,
                "cy": (2 * y1 + b_top + b_bot) / 2 / h,
                "box": (nx1, ny1, nx2, ny2),
            })
        return results

    def _switch_location(self, target_key: str):
        locale = self._get_locale()
        loc_names = _LOCATIONS.get(locale, _LOCATIONS["zh_CN"])
        if target_key not in loc_names:
            raise ValueError(f"未知 location key: {target_key}")

        # Step 1: Click the Switch Depot button (white pill right of the
        # depot title). Measured: pill (0.445-0.562, 0.166-0.208), text
        # "Switch Depot" at (0.455-0.540, 0.180-0.210).
        switch_text = self.lang.WarehouseTransferTask.k_3cb6baa6
        self.log_info(f"[switch] clicking Switch Depot button, locale={locale}")
        clicked_btn = False
        self.next_frame()
        frame = self.executor.frame
        if frame is not None:
            h, w = frame.shape[:2]
            btn_box = (int(w * 0.455), int(h * 0.180), int(w * 0.540), int(h * 0.210))
            text = ocr_text(frame, box=btn_box, psm=7)
            self._save_switch_debug("switch_btn", frame,
                                    boxes=[(btn_box, f"Tess: '{text}'")],
                                    dot=(int(0.50 * w), int(0.193 * h)))
            if ocr_match(frame, btn_box, switch_text):
                self.log_info(f"[switch] Tesseract found '{text}', clicking")
                self.click_relative(0.50, 0.193)
                clicked_btn = True
        if not clicked_btn:
            btn = self.wait_ocr(
                box=self.box_of_screen(0.44, 0.16, 0.56, 0.21, name="switch_btn_area"),
                match=switch_text,
                time_out=2,
                raise_if_not_found=False,
            )
            if btn:
                self.click(btn[0])
            else:
                self.log_info("[switch] OCR missed, using positional click")
                self.click_relative(0.50, 0.193)
        self.sleep(1.0)

        # Step 2: Click the target depot row in the modal
        target_text = loc_names[target_key]
        self.log_info(f"[switch] selecting depot '{target_text}'")

        clicked_target = False
        self.next_frame()
        frame = self.executor.frame
        if frame is not None:
            h, w = frame.shape[:2]
            rows = self._find_depot_rows(frame)
            dbg_boxes = [(r["box"], f"Tess: '{r['text']}'") for r in rows]
            dot = None
            for r in rows:
                if target_text.lower() in r["text"].lower():
                    self.log_info(f"[switch] Tesseract found '{r['text']}' at y={r['cy']:.3f}")
                    dot = (int(r["cx"] * w), int(r["cy"] * h))
                    break
            self._save_switch_debug("switch_modal", frame, boxes=dbg_boxes, dot=dot)
            if dot is not None:
                self.click_relative(dot[0] / w, dot[1] / h)
                clicked_target = True

        if not clicked_target:
            # Fallback: framework OCR over the modal options area
            option = self.wait_ocr(
                box=self.box_of_screen(0.35, 0.35, 0.80, 0.68, name="switch_menu"),
                match=target_text,
                time_out=2,
                raise_if_not_found=False,
            )
            if option:
                self.click(option[0])
                clicked_target = True

        if not clicked_target:
            # Positional fallback: rows in the initial (unselected) modal
            # state sit at y=0.418 (Wuling) / 0.510 (Valley IV)
            _OPTION_POSITIONS = {"wuling": 0.418, "valley4": 0.510}
            option_y = _OPTION_POSITIONS.get(target_key, 0.418)
            self.log_info(f"[switch] OCR missed, positional click y={option_y}")
            if frame is not None:
                h, w = frame.shape[:2]
                self._save_switch_debug("switch_modal_fallback", frame,
                                        dot=(int(0.575 * w), int(option_y * h)))
            self.click_relative(0.575, option_y)
        self.sleep(0.5)

        # Step 3: Click Confirm if it appears
        self._maybe_click_confirm()

        # Step 4: Wait for "Connected" on the modal's bottom button (the
        # yellow Confirm pill turns into a greyed "Connected" pill)
        connected_text = self.lang.WarehouseTransferTask.k_65fe35c4
        self.log_info(f"[switch] waiting for '{connected_text}'...")
        connected = False
        for i in range(15):
            self.next_frame()
            frame = self.executor.frame
            if frame is not None:
                h, w = frame.shape[:2]
                l, t, r, b = self._MODAL_BTN_TEXT_BOX
                conn_box = (int(w * l), int(h * t), int(w * r), int(h * b))
                if ocr_match(frame, conn_box, connected_text):
                    self.log_info(f"[switch] Tesseract detected '{connected_text}' after {i} polls")
                    connected = True
                    break
            # Also try framework OCR
            hits = self.ocr(box=self.box_of_screen(0.70, 0.77, 0.88, 0.845), match=connected_text, threshold=0.1)
            if hits:
                self.log_info(f"[switch] framework OCR detected connected after {i} polls")
                connected = True
                break
            self.sleep(0.5)
        if not connected:
            self.log_info("[switch] neither OCR detected 'Connected', waiting 5s")
            self.sleep(5.0)

        # Step 5: Close the Switch Depot modal
        self.sleep(1.0)
        self._close_switch_depot_modal()
        self.log_info("[switch] depot switch complete")

    def _close_switch_depot_modal(self):
        """Close the Switch Depot modal by clicking its X button."""
        self.log_info("[close_modal] clicking X button")
        # X button at top-right of modal: measured (0.858, 0.186)
        frame = self.executor.frame
        if frame is not None:
            h, w = frame.shape[:2]
            self._save_switch_debug("close_modal", frame,
                                    dot=(int(0.858 * w), int(0.186 * h)))
        # SINGLE click only. The Backpack panel's own X close button sits at
        # the SAME position (measured 0.87, 0.184 on r0_cell_0_0.png) once
        # the modal is gone - a blind second click closed the entire depot
        # UI, which broke Quick Stash and everything after (2026-08-13 log).
        self.click_relative(0.858, 0.186)
        self.sleep(0.8)

    # Grid layout constants (relative to full 3840x2160 game frame).
    # Initial estimates - refined at runtime by _measure_grid_from_contours().
    _GRID_LEFT = 0.1242
    _GRID_TOP = 0.2986
    _GRID_RIGHT = 0.5542
    _GRID_BOTTOM = 0.6784
    _GRID_COLS = 8
    _GRID_ROWS = 4  # visible rows before scrolling
    _measured_row_height_px: int | None = None  # set by contour measurement

    def _hover_absolute(self, px_x: int, px_y: int):
        """Move the real cursor AND send WM_MOUSEMOVE to trigger game tooltip."""
        interaction = self.executor.interaction
        interaction.try_activate()
        abs_x, abs_y = interaction.capture.get_abs_cords(px_x, px_y)
        win32api.SetCursorPos((abs_x, abs_y))
        lParam = win32api.MAKELONG(px_x, px_y)
        win32gui.PostMessage(interaction.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)

    def _scroll_precise(self, px_x: int, px_y: int, delta: int):
        """Send WM_MOUSEWHEEL with exact delta for pixel-precise scrolling.

        Args:
            px_x, px_y: cursor position (client coords)
            delta: wheel delta (negative = scroll down). 120 = 1 standard notch.
        """
        interaction = self.executor.interaction
        interaction.try_activate()
        abs_x, abs_y = interaction.capture.get_abs_cords(px_x, px_y)
        win32api.SetCursorPos((abs_x, abs_y))
        wParam = (delta & 0xFFFF) << 16
        lParam = win32api.MAKELONG(abs_x, abs_y)
        win32gui.PostMessage(interaction.hwnd, 0x020A, wParam, lParam)

    @staticmethod
    def _extract_tooltip_text(crop: np.ndarray) -> tuple[str, str]:
        """Detect tooltip panel via color, crop to text regions, OCR.

        Verified against 256 tooltip crops (8 rounds x 32 cells) from actual
        4K gameplay screenshots: 0 missed titles, 8/8 container descs.

        Returns (title_text, desc_text).
        """
        h, w = crop.shape[:2]
        if h < 20 or w < 50:
            return "", ""

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # Panel mask: sat < 120 because high-rarity tooltips have an
        # orange/tan tinted panel (measured sat_med 34-86, sat_p90 107)
        # while the common grey panel is sat < 45. Background cells below
        # the panel stay val > 95 so the val bound still excludes them.
        panel_mask = (val > 25) & (val < 95) & (sat < 120)
        row_coverage = panel_mask.sum(axis=1) / w

        # Find panel rows: need >35% coverage
        panel_rows = np.where(row_coverage > 0.35)[0]
        if len(panel_rows) < 10:
            # Fallback: try the top 33% band directly with Tesseract
            band = crop[0:int(h * 0.33), :]
            if band.size == 0:
                return "", ""
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            inv = 255 - gray
            _, binary = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            padded = cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
            title_text = ocr_text(padded, psm=7)
            title_text = title_text.strip().split('\n')[0].strip()
            return title_text, ""

        # Find contiguous panel block from first panel row
        panel_top = panel_rows[0]
        panel_bottom = panel_top
        for i in range(1, len(panel_rows)):
            if panel_rows[i] - panel_rows[i - 1] > 15:
                break
            panel_bottom = panel_rows[i]

        # Text mask: white pixels within panel region only
        text_mask = (val > 170) & (sat < 60)
        text_mask[:panel_top, :] = False   # ignore above panel
        text_mask[panel_bottom:, :] = False  # ignore below panel

        # Cluster on the LEFT 40% of columns only: bright item-icon
        # highlights (bottles, metallic towers) to the right of the panel
        # otherwise bridge row gaps and merge title/category/icon into one
        # giant cluster. The title always starts in the left region; its
        # full width is measured later against the full-width mask.
        left_mask = text_mask.copy()
        left_mask[:, int(w * 0.4):] = False

        text_row_counts = left_mask.sum(axis=1)
        text_rows = np.where(text_row_counts > 5)[0]
        if len(text_rows) == 0:
            return "", ""

        clusters = []
        cluster_start = text_rows[0]
        for i in range(1, len(text_rows)):
            if text_rows[i] - text_rows[i - 1] > 8:
                clusters.append((cluster_start, text_rows[i - 1]))
                cluster_start = text_rows[i]
        clusters.append((cluster_start, text_rows[-1]))

        # Title = first cluster that is tall enough (>=15px) and starts in
        # the left 40% of the crop. Rejects spurious clusters from
        # neighboring-cell count digits (short, right-aligned white blobs).
        title_cluster = None
        for c_top, c_bot in clusters:
            if c_bot - c_top < 15:
                continue
            c_cols = np.where(text_mask[c_top:c_bot + 1, :].sum(axis=0) > 0)[0]
            if len(c_cols) == 0 or c_cols[0] > w * 0.4:
                continue
            title_cluster = (c_top, c_bot)
            break
        if title_cluster is None:
            return "", ""
        title_top, title_bottom = title_cluster
        title_top = max(0, title_top - 2)
        # +12: g/y descenders extend 11-12px below the cluster end (cluster
        # rows need >5px in the left 40%; descender-only rows have <=5px so
        # the cluster stops at the baseline, which made Tesseract read
        # 'Mining Rig' as 'Minina Ria'). Measured descender end 67-68 vs
        # cluster end 56, category text starts >=79 (r0_tooltip_3_7,
        # r4_tooltip_0_2, r5_tooltip_1_3).
        title_bottom = min(h, title_bottom + 12)

        # Horizontal bounds with gap detection (excludes adjacent cell content)
        col_has_text = text_mask[title_top:title_bottom, :].sum(axis=0) > 0
        title_cols = np.where(col_has_text)[0]
        if len(title_cols) == 0:
            return "", ""
        title_left = max(0, title_cols[0] - 5)
        title_right = title_cols[-1] + 5
        last_col = title_cols[0]
        for i in range(1, len(title_cols)):
            if title_cols[i] - title_cols[i - 1] > 30:
                title_right = last_col + 5
                break
            last_col = title_cols[i]
        else:
            title_right = min(w, title_cols[-1] + 5)

        # OCR title
        title_crop = crop[title_top:title_bottom, title_left:title_right]
        title_gray = cv2.cvtColor(title_crop, cv2.COLOR_BGR2GRAY)
        title_inv = 255 - title_gray
        _, title_bin = cv2.threshold(title_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        title_padded = cv2.copyMakeBorder(title_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
        title_text = ocr_text(title_padded, psm=7)
        title_text = title_text.strip().split('\n')[0].strip()
        # Trailing Roman numerals: Tesseract reads 'I' as '|' and 'II' as
        # 'Il' ('Aerospace Material Il', 'Marsh Gas Mk |'). A trailing word
        # made only of I/l/| chars can only be a Roman numeral.
        title_text = re.sub(
            r' ([Il|]+)$', lambda m: ' ' + 'I' * len(m.group(1)), title_text)

        # Description ("Filled with X") sits in a DARKER band below the
        # panel (val_med ~18 vs panel ~37) with dim grey text (val ~115-160)
        # that fails the val>170 title mask. Detect the dark band by
        # val<30 row coverage and OCR it with a dim-text mask.
        desc_text = ""
        dark_cov = (val < 30).sum(axis=1) / w
        band_top = None
        band_bottom = None
        for r in range(panel_bottom + 1, min(h, panel_bottom + 12)):
            if dark_cov[r] > 0.4:
                band_top = r
                break
        if band_top is not None:
            band_bottom = band_top
            for r in range(band_top + 1, h):
                if dark_cov[r] > 0.4:
                    band_bottom = r
                else:
                    break
        if band_top is not None and band_bottom - band_top > 20:
            dim_mask = (val > 110) & (sat < 60)
            dim_mask[:band_top, :] = False
            dim_mask[band_bottom + 1:, :] = False
            dim_rows = np.where(dim_mask.sum(axis=1) > 5)[0]
            if len(dim_rows) >= 8:
                d_top = max(band_top, dim_rows[0] - 2)
                d_bot = min(band_bottom, dim_rows[-1] + 2)
                d_cols = np.where(dim_mask[d_top:d_bot + 1, :].sum(axis=0) > 0)[0]
                if len(d_cols) > 10:
                    dl = max(0, d_cols[0] - 5)
                    dr = min(w, d_cols[-1] + 5)
                    desc_crop = crop[d_top:d_bot + 1, dl:dr]
                    desc_gray = cv2.cvtColor(desc_crop, cv2.COLOR_BGR2GRAY)
                    desc_inv = 255 - desc_gray
                    _, desc_bin = cv2.threshold(desc_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    desc_padded = cv2.copyMakeBorder(desc_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                    desc_text = ocr_text(desc_padded, psm=7)
                    desc_text = desc_text.strip().split('\n')[0].strip()
                    if sum(ch.isalpha() for ch in desc_text) < 4:
                        desc_text = ""

        return title_text, desc_text

    @staticmethod
    def _count_ocr_cell(frame, cx_px: int, cy_px: int, half_cw: int, half_ch: int) -> str:
        """OCR the item-count text of one grid cell from a clean (no-tooltip) frame.

        The count sits in a fixed band above the rarity bar: rows
        [y_bot-62, y_bot-13] relative to cell bottom (measured at 4K). White
        digits (val>170, sat<60) are isolated via HSV mask, then icon-highlight
        noise is removed with connected components: digit glyphs are 15-45px
        tall with bottoms on a shared baseline; period glyphs are 3-12px
        squares sitting on the same baseline. The kept components are rendered
        as a clean binary, upscaled 3x and padded 30px (Tesseract drops
        periods/digits with less padding).

        psm 7 is primary; when its output length disagrees with the kept
        component count (each component = one glyph) psm 8 is tried - psm 7
        returns '' on lone digits ('8') and truncates '27'->'2', while psm 8
        alone hallucinates digits on K-values ('12K'->'412K').
        Verified on all 96 cells of rounds r0-r2 (zero empties, all periods).
        """
        whitelist = "-c tessedit_char_whitelist=0123456789.KM"
        y_bot = cy_px + half_ch
        band = frame[max(0, y_bot - 62):y_bot - 13, max(0, cx_px - half_cw):cx_px + half_cw]
        if band.size == 0:
            return ""
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 2] > 170) & (hsv[:, :, 1] < 60)).astype(np.uint8)
        if mask.sum() < 30:
            return ""
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        bh = band.shape[0]
        cand = [i for i in range(1, n)
                if 15 <= stats[i][3] <= 45 and stats[i][1] + stats[i][3] >= bh - 18]
        if not cand:
            return ""
        baseline = int(np.median([stats[i][1] + stats[i][3] for i in cand]))
        keep = []
        for i in range(1, n):
            x, y, w_, h_, area = stats[i]
            if abs((y + h_) - baseline) > 6:
                continue
            if 15 <= h_ <= 45:
                keep.append(i)  # digit / K / M
            elif 3 <= h_ <= 12 and 3 <= w_ <= 12 and area >= 6:
                keep.append(i)  # decimal period on the baseline
        if not keep:
            return ""
        km = np.isin(labels, keep)
        rows = np.where(km.sum(axis=1) > 0)[0]
        cols = np.where(km.sum(axis=0) > 0)[0]
        t = max(0, rows[0] - 3)
        b = min(bh, rows[-1] + 4)
        l = max(0, cols[0] - 4)
        r = min(band.shape[1], cols[-1] + 5)
        binimg = np.where(km[t:b, l:r], 0, 255).astype(np.uint8)
        up = cv2.resize(binimg, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        up = cv2.copyMakeBorder(up, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        txt = ocr_text(up, psm=7, extra_config=whitelist).strip().rstrip(".")
        if len(txt) != len(keep):
            txt8 = ocr_text(up, psm=8, extra_config=whitelist).strip().rstrip(".")
            if len(txt8) == len(keep) or not txt:
                txt = txt8
        return txt

    @staticmethod
    def _parse_count(text: str) -> int:
        """Parse item count from OCR text like '23', '80K', '7.74K', '1.2M'."""
        if not text:
            return 0
        text = text.strip().upper().replace(',', '').replace(' ', '')
        text = re.sub(r'[^0-9.KM]', '', text)
        if not text:
            return 0
        try:
            if text.endswith('K'):
                return int(float(text[:-1]) * 1000)
            elif text.endswith('M'):
                return int(float(text[:-1]) * 1000000)
            else:
                return int(float(text))
        except (ValueError, IndexError):
            return 0

    def _build_match_targets(self, item_key: str) -> list[str]:
        """Build list of lowercase target strings to match against OCR output."""
        targets = [item_key.lower()]
        game_en = ITEM_GAME_ENGLISH.get(item_key, "")
        if game_en:
            targets.append(game_en.lower())
        en_code = ITEM_TRANSLATION_DICT.get(item_key, "")
        if en_code:
            targets.append(en_code.replace("_", " ").lower())
        en_to_zh = {v.lower(): k for k, v in ITEM_GAME_ENGLISH.items()}
        if item_key.lower() in en_to_zh:
            zh_key = en_to_zh[item_key.lower()]
            targets.append(zh_key)
            code = ITEM_TRANSLATION_DICT.get(zh_key, "")
            if code:
                targets.append(code.replace("_", " ").lower())
        return targets


    def _scan_row_cells(self, row: int, scroll_round: int, debug_prefix: str = "r") -> list[dict]:
        """Hover every cell in one grid row; OCR tooltip title/desc + count.

        Extracted verbatim from the verified single-item scan pipeline.
        Returns one dict per captured cell (cells whose frame grab failed
        are skipped; cells with no tooltip text have title ""):
          {"col", "cx", "cy", "title", "desc", "count"}
        """
        w, h = self.width, self.height
        col_width = (self._GRID_RIGHT - self._GRID_LEFT) / self._GRID_COLS
        row_height = (self._GRID_BOTTOM - self._GRID_TOP) / self._GRID_ROWS
        half_cw = int(col_width * w / 2)
        half_ch = int(row_height * h / 2)
        debug_dir = self._debug_dir()

        # Phase 1: Hover each cell to capture tooltip
        row_captures = []
        for col in range(self._GRID_COLS):
            cx_px = int((self._GRID_LEFT + col_width * (col + 0.5)) * w)
            cy_px = int((self._GRID_TOP + row_height * (row + 0.5)) * h)
            self._hover_absolute(cx_px, cy_px)
            self.sleep(0.65)
            self.next_frame()
            frame = self.executor.frame
            if frame is None:
                continue
            tx1 = min(w, cx_px + 48)
            tx2 = min(w, cx_px + 700)
            ty1 = max(0, cy_px + 15)
            ty2 = min(h, cy_px + 225)
            row_captures.append((col, cx_px, cy_px, frame.copy(), (tx1, ty1, tx2, ty2)))

        # Phase 2: Move cursor away from grid, capture clean frame for counts
        safe_x = int(0.75 * w)  # backpack area (right side)
        safe_y = int(0.50 * h)
        self._hover_absolute(safe_x, safe_y)
        self.sleep(0.4)
        self.next_frame()
        clean_frame = self.executor.frame
        if clean_frame is None:
            clean_frame = row_captures[-1][3] if row_captures else None

        # Move cursor back to grid center (scroll requires cursor in grid area)
        grid_cx = int((self._GRID_LEFT + (self._GRID_RIGHT - self._GRID_LEFT) / 2) * w)
        grid_cy = int((self._GRID_TOP + (self._GRID_BOTTOM - self._GRID_TOP) / 2) * h)
        self._hover_absolute(grid_cx, grid_cy)
        self.sleep(0.15)

        # Phase 3: Read counts from clean frame (no tooltip overlay)
        row_counts = {}
        if clean_frame is not None:
            for col, cx_px, cy_px, _, _ in row_captures:
                count_str = self._count_ocr_cell(clean_frame, cx_px, cy_px, half_cw, half_ch)
                row_counts[col] = self._parse_count(count_str)

        # Phase 4: OCR tooltips for item names
        results = []
        for col, cx_px, cy_px, frame, (tx1, ty1, tx2, ty2) in row_captures:
            # OCR tooltip via preprocessing (color-based text isolation)
            tooltip_crop = frame[ty1:ty2, tx1:tx2]
            title_text, desc_text = self._extract_tooltip_text(tooltip_crop)

            try:
                debug_frame = frame.copy()
                cv2.rectangle(debug_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)
                cv2.circle(debug_frame, (cx_px, cy_px), 12, (0, 0, 255), -1)
                cv2.imwrite(str(debug_dir / f"{debug_prefix}{scroll_round}_cell_{row}_{col}.png"), debug_frame)
                if tooltip_crop.size > 0:
                    cv2.imwrite(str(debug_dir / f"{debug_prefix}{scroll_round}_tooltip_{row}_{col}.png"), tooltip_crop)
                # Save the preprocessed title image that was actually sent to OCR
                if title_text:
                    h_c, w_c = tooltip_crop.shape[:2]
                    hsv_c = cv2.cvtColor(tooltip_crop, cv2.COLOR_BGR2HSV)
                    val_c = hsv_c[:, :, 2]
                    sat_c = hsv_c[:, :, 1]
                    t_mask = (val_c > 170) & (sat_c < 60)
                    t_rows = np.where(t_mask.sum(axis=1) > 5)[0]
                    if len(t_rows) > 0:
                        t_top = max(0, t_rows[0] - 2)
                        t_bot = min(h_c, t_rows[0] + 40)
                        title_debug = tooltip_crop[t_top:t_bot, :]
                        cv2.imwrite(str(debug_dir / f"{debug_prefix}{scroll_round}_title_{row}_{col}.png"), title_debug)
            except Exception:
                pass

            item_count = row_counts.get(col, 0)

            if not title_text and col == 0 and scroll_round == 0:
                self.log_info(f"[grid_scan] ({row},{col}) empty! diag={_ocr_diag}")

            results.append({"col": col, "cx": cx_px, "cy": cy_px,
                            "title": title_text, "desc": desc_text, "count": item_count})
        return results

    def _scroll_grid_page(self):
        """Scroll the item grid down by one page (~4 rows)."""
        w, h = self.width, self.height
        scroll_x = int((self._GRID_LEFT + (self._GRID_RIGHT - self._GRID_LEFT) / 2) * w)
        scroll_y = int((self._GRID_TOP + (self._GRID_BOTTOM - self._GRID_TOP) / 2) * h)
        self._hover_absolute(scroll_x, scroll_y)
        self.sleep(0.2)
        self._scroll_precise(scroll_x, scroll_y, -230)
        self.sleep(1.2)
        self.next_frame()
        self.sleep(0.3)

    def _scan_grid_for_item(self, item_key: str):
        """Scan item grid by hovering each cell, reading tooltip via OCR.

        Returns a Box at the bottom-right-most match, or None. The count
        of the matched cell is exposed as self._last_scan_count.
        """
        targets = self._build_match_targets(item_key)
        w, h = self.width, self.height
        col_width = (self._GRID_RIGHT - self._GRID_LEFT) / self._GRID_COLS
        row_height = (self._GRID_BOTTOM - self._GRID_TOP) / self._GRID_ROWS
        half_cw = int(col_width * w / 2)
        half_ch = int(row_height * h / 2)

        seen_items = set()
        last_top_left_text = None
        last_match = None
        last_match_count = 0
        best_score = 0.0
        self._last_scan_count = 0

        self.log_info(f"[grid_scan] targets={targets}, grid={self._GRID_COLS}x{self._GRID_ROWS}")
        if not _tess_mod._initialized:
            self.log_info("[grid_scan] WARNING: tesseract not initialized!")

        for scroll_round in range(8):
            self.log_info(f"[grid_scan] --- round {scroll_round} ---")
            top_left_text = None

            for row in range(self._GRID_ROWS):
                for cell in self._scan_row_cells(row, scroll_round):
                    col = cell["col"]
                    title_text, desc_text = cell["title"], cell["desc"]
                    item_count = cell["count"]

                    if not title_text:
                        continue

                    full_text = title_text if not desc_text else f"{title_text} | {desc_text}"
                    count_label = f" x{item_count}" if item_count else ""
                    self.log_info(f"[grid_scan] ({row},{col}) -> '{full_text}'{count_label}")
                    seen_items.add(title_text.lower())

                    if row == 0 and col == 0 and top_left_text is None:
                        top_left_text = title_text

                    match_text = full_text.lower()
                    score = max(item_match_score(match_text, t, _MATCH_VOCAB) for t in targets)
                    # >= keeps the bottom-right-most cell among equal scores
                    # (stacks fill top-left first, so the last cell is the
                    # partial stack the game would deplete first).
                    if score > 0.0 and score >= best_score:
                        best_score = score
                        last_match = Box(cell["cx"] - half_cw, cell["cy"] - half_ch,
                                         int(col_width * w), int(row_height * h), name=title_text)
                        last_match_count = item_count
                        self.log_info(f"[grid_scan] MATCH at ({row},{col}) score={score:.3f}{count_label}")

            if last_match:
                self.log_info(f"[grid_scan] returning: '{last_match.name}' x{last_match_count} "
                              f"score={best_score:.3f}")
                self._last_scan_count = last_match_count
                return last_match

            if top_left_text and top_left_text == last_top_left_text:
                self.log_info(f"[grid_scan] end of list (top-left unchanged: '{top_left_text}')")
                break
            last_top_left_text = top_left_text

            self._scroll_grid_page()

        self.log_info(f"[grid_scan] NOT found. Seen: {sorted(seen_items)}")
        return None

    def _ctrl_click(self, box):
        win32api.keybd_event(
            win32con.VK_CONTROL, 0, 0, 0
        )  # 确认使用send_key：ctrl为系统修饰键，用于ctrl+点击多选，非游戏可配置热键
        try:
            self.sleep(0.03)
            self.click(box,   down_time=0.03, after_sleep=0, key="left")
            self.sleep(0.03)
        finally:
            win32api.keybd_event(win32con.VK_CONTROL, 0, win32con.KEYEVENTF_KEYUP, 0)
        self.sleep(0.15)

    # Backpack panel (right side of depot UI), measured on 4K screenshots:
    # grid region x 2300-3398, y 639-1499; cell pitch 206.5px (rel 0.0538*w).
    # Occupied cells are found via their saturated rarity bar (sat>80 &
    # val>80) - scene fog is bright but DESATURATED, so this mask is
    # fog-immune where brightness masks blow out. Verified identical
    # results on 4 frames including heavy fog.
    _BACKPACK_REGION = (0.599, 0.296, 0.885, 0.694)

    def _deposit_backpack_item(self) -> bool:
        """Ctrl-click the bottom-right-most occupied backpack cell.

        Used instead of Quick Stash, which silently fails for some items.
        Deposited items appear appended after existing backpack items, so
        the bottom-right-most occupied cell is the item just withdrawn.
        Returns True if an occupied cell was found and clicked.
        """
        self.next_frame()
        frame = self.executor.frame
        if frame is None:
            return False
        h, w = frame.shape[:2]
        l, t, r, b = self._BACKPACK_REGION
        x0, y0 = int(l * w), int(t * h)
        x1, y1 = int(r * w), int(b * h)
        region = frame[y0:y1, x0:x1]
        if region.size == 0:
            return False
        cell = int(0.0538 * w)

        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        mask = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 80)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        best = None  # (row_bucket, x, cx, cy)
        boxes = []
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            # Rarity bar spans the cell width; icon may merge with it but
            # the component bbox bottom stays at the cell bottom.
            if area < 2000 or not (0.7 * cell <= bw <= 1.1 * cell):
                continue
            bot = by + bh
            cx = x0 + bx + bw // 2
            cy = y0 + bot - cell // 2
            boxes.append(((x0 + bx, y0 + by, x0 + bx + bw, y0 + bot), ""))
            key = (bot // (cell // 2), bx)
            if best is None or key > best[0]:
                best = (key, cx, cy)

        if best is None:
            self._save_switch_debug("deposit_backpack", frame,
                                    boxes=[((x0, y0, x1, y1), "no occupied cell")])
            return False

        _, cx, cy = best
        self._save_switch_debug("deposit_backpack", frame, boxes=boxes, dot=(cx, cy))
        self.log_info(f"[deposit] ctrl-clicking backpack item at ({cx},{cy})")
        self._ctrl_click(Box(cx - cell // 2, cy - cell // 2, cell, cell, name="backpack_item"))
        return True

    def _do_deposit(self):
        """Deposit the withdrawn item at the current (destination) depot.

        Ctrl-clicks the backpack item - Quick Stash silently fails for
        some items (game-side issue). Fallbacks: fixed bottom-right
        backpack cell position, then the Quick Stash button (OCR-gated).
        """
        self.log_info("[store] depositing via backpack ctrl-click")
        if not self._deposit_backpack_item():
            # Fallback 1: fixed position of bottom-right backpack cell
            # (user-measured red dot, rel 0.8430/0.6301 at 4K).
            self.log_info("[store] detection failed, ctrl-clicking fixed position")
            cell = int(0.0538 * self.width)
            cx = int(0.8430 * self.width)
            cy = int(0.6301 * self.height)
            self._ctrl_click(Box(cx - cell // 2, cy - cell // 2, cell, cell,
                                 name="backpack_fixed"))
            # Fallback 2: Quick Stash button (works for most items).
            # Pill measured on r0_cell_0_0.png (4K): x 0.632-0.739,
            # y 0.699-0.738.
            store_text = self.lang.WarehouseTransferTask.k_d661f6da
            self.next_frame()
            frame = self.executor.frame
            if frame is not None:
                h, w = frame.shape[:2]
                store_box = (int(w * 0.625), int(h * 0.695), int(w * 0.75), int(h * 0.745))
                if ocr_match(frame, store_box, store_text):
                    self.log_info("[store] fallback: clicking Quick Stash")
                    self.click_relative(0.67, 0.718)
        # Deposit usually has NO confirm dialog - never blind-click
        self._maybe_click_confirm(positional_fallback=False)

    def run(self):
        self.ensure_main()
        try:
            diag = ensure_tesseract()
            self.log_info(f"[run] Tesseract: {diag}")
        except Exception as e:
            self.log_info(f"[run] Tesseract FAILED: {e}")

        from_key = str(self.config.get("发货仓库", "valley4")).strip()
        to_key = str(self.config.get("收货仓库", "wuling")).strip()
        if from_key == to_key:
            raise RuntimeError("Source and destination depot cannot be the same")

        item_config = str(self.config.get("物品", "")).strip()
        if not item_config:
            raise RuntimeError("No item selected")

        # Resolve English game name to Chinese key (for category navigation)
        # Build reverse mapping: game_english -> chinese_key
        _en_to_zh = {v.lower(): k for k, v in ITEM_GAME_ENGLISH.items()}
        item_key_zh = _en_to_zh.get(item_config.lower(), item_config)
        # item_key for scan: use whatever was configured (works with both EN and ZH)
        item_key = item_key_zh

        max_times = int(self.config.get("转移轮次", 10))
        locale = self._get_locale()
        self.log_info(f"[run] from={from_key}, to={to_key}, item={item_config} (zh={item_key_zh}), "
                      f"rounds={max_times}, locale={locale}, resolution={self.width}x{self.height}")
        self.press_key("b")
        self.sleep(2.0)
        # Wait for depot UI to appear
        self.wait_until(
            lambda: self._detect_current_location() is not None,
            time_out=5,
            raise_if_not_found=False,
        )
        while True:
            # Always check current depot and switch if needed
            current = self._detect_current_location()
            if current != from_key:
                self.log_info(f"[run] current depot={current}, switching to source={from_key}")
                self._switch_location(from_key)
            self._to_one_type_page(item_key)
            self.log_info(f"[run] scanning for: {item_config}")

            # Find item by grid-scanning with hover + OCR
            icon = self._scan_grid_for_item(item_key)

            if not icon:
                self.log_info(f"[run] item not found: {item_config}")
                break
            self._ctrl_click(icon)
            self.sleep(0.35)

            self.log_info(f"[run] switching to destination={to_key}")
            self._switch_location(to_key)

            self._do_deposit()
            max_times -= 1
            if max_times <= 0:
                break
            self.log_info(f"切回发货仓库={from_key}")
            self._switch_location(from_key)
        self.log_info("仓库转移任务完成")
