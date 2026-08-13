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
from src.ocr import ocr_text, ocr_match, ocr_frame, ensure_tesseract
from src.ocr import tesseract_ocr as _tess_mod
from src.ocr.tesseract_ocr import _ocr_diag
from src.tasks.onetime.item_matcher import matches as item_matches

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

        # Primary: Tesseract OCR on the title region
        if frame is not None:
            h, w = frame.shape[:2]
            # Title area: "Valley IV Depot" / "武陵仓库" at ~(0.10, 0.17) to (0.29, 0.23)
            title_box = (int(w * 0.10), int(h * 0.17), int(w * 0.29), int(h * 0.23))
            text = ocr_text(frame, box=title_box, psm=6)
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

    def _maybe_click_confirm(self):
        """Click Confirm button if it appears (Tesseract primary, positional fallback)."""
        confirm_text = self.lang.WarehouseTransferTask.k_b56d9ac6
        self.next_frame()
        frame = self.executor.frame

        # Try Tesseract first
        if frame is not None:
            h, w = frame.shape[:2]
            confirm_box = (int(w * 0.78), int(h * 0.84), int(w * 0.97), int(h * 0.97))
            if ocr_match(frame, confirm_box, confirm_text):
                self.log_info(f"[confirm] Tesseract found '{confirm_text}', clicking")
                self.click_relative(0.88, 0.91)
                self.sleep(0.5)
                return True

        # Try framework OCR
        hits = self.wait_ocr(
            box=self.box_of_screen(0.78, 0.84, 0.97, 0.97, name="confirm_btn_area"),
            match=confirm_text,
            time_out=2,
            raise_if_not_found=False,
        )
        if hits:
            self.log_info(f"[confirm] framework OCR found, clicking")
            self.click(hits[0])
            self.sleep(0.5)
            return True

        # Positional fallback
        self.log_info("[confirm] OCR missed, clicking confirm position")
        self.click_relative(0.88, 0.91)
        self.sleep(0.5)
        return True

    def _switch_location(self, target_key: str):
        locale = self._get_locale()
        loc_names = _LOCATIONS.get(locale, _LOCATIONS["zh_CN"])
        if target_key not in loc_names:
            raise ValueError(f"未知 location key: {target_key}")

        # Step 1: Click the Switch Depot button (teal pill in depot header)
        switch_text = self.lang.WarehouseTransferTask.k_3cb6baa6
        self.log_info(f"[switch] clicking Switch Depot button, locale={locale}")
        btn = self.wait_ocr(
            box=self.box_of_screen(0.20, 0.03, 0.35, 0.10, name="switch_btn_area"),
            match=switch_text,
            time_out=2,
            raise_if_not_found=False,
        )
        if btn:
            self.click(btn[0])
        else:
            self.log_info("[switch] framework OCR missed, using positional click")
            self.click_relative(0.26, 0.06)
        self.sleep(1.0)

        # Step 2: Click the target depot in the modal
        # Positions: Wuling ~(0.35, 0.14), Valley IV ~(0.35, 0.18)
        _OPTION_POSITIONS = {"wuling": 0.14, "valley4": 0.18}
        target_text = loc_names[target_key]
        self.log_info(f"[switch] selecting depot '{target_text}'")

        # Try Tesseract to find the target depot name in modal
        clicked_target = False
        self.next_frame()
        frame = self.executor.frame
        if frame is not None:
            h, w = frame.shape[:2]
            modal_box = (int(w * 0.15), int(h * 0.08), int(w * 0.55), int(h * 0.30))
            detections = ocr_frame(frame, box=modal_box, psm=6)
            for det in detections:
                if target_text.lower() in det["text"].lower():
                    # Click center of detected text
                    cx = det["x"] + det["w"] // 2
                    cy = det["y"] + det["h"] // 2
                    self.log_info(f"[switch] Tesseract found '{det['text']}' at ({cx},{cy})")
                    self.click_relative(cx / w, cy / h)
                    clicked_target = True
                    break

        if not clicked_target:
            # Fallback: framework OCR
            option = self.wait_ocr(
                box=self.box_of_screen(0.15, 0.08, 0.55, 0.30, name="switch_menu"),
                match=target_text,
                time_out=2,
                raise_if_not_found=False,
            )
            if option:
                self.click(option[0])
                clicked_target = True

        if not clicked_target:
            option_y = _OPTION_POSITIONS.get(target_key, 0.16)
            self.log_info(f"[switch] OCR missed, positional click y={option_y}")
            self.click_relative(0.35, option_y)
        self.sleep(0.5)

        # Step 3: Click Confirm if it appears
        self._maybe_click_confirm()

        # Step 4: Wait for "Connected" using Tesseract
        connected_text = self.lang.WarehouseTransferTask.k_65fe35c4
        self.log_info(f"[switch] waiting for '{connected_text}'...")
        connected = False
        for i in range(15):
            self.next_frame()
            frame = self.executor.frame
            if frame is not None:
                h, w = frame.shape[:2]
                # Connected button at bottom-right of modal ~(0.78, 0.84) to (0.97, 0.97)
                conn_box = (int(w * 0.78), int(h * 0.84), int(w * 0.97), int(h * 0.97))
                if ocr_match(frame, conn_box, connected_text):
                    self.log_info(f"[switch] Tesseract detected '{connected_text}' after {i} polls")
                    connected = True
                    break
            # Also try framework OCR
            hits = self.ocr(box=self.box_of_screen(0.78, 0.84, 0.97, 0.97), match=connected_text, threshold=0.1)
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
        # X button at top-right of modal: ~(0.54, 0.06)
        self.click_relative(0.54, 0.06)
        self.sleep(0.8)
        # Click again in case first click missed
        self.click_relative(0.54, 0.06)
        self.sleep(0.5)

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

        Uses HSV analysis to find the dark-grey tooltip panel, then isolates
        white text within it. Skips description OCR if no text exists there.

        Returns (title_text, desc_text).
        """
        h, w = crop.shape[:2]
        if h < 20 or w < 50:
            return "", ""

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        val = hsv[:, :, 2]
        sat = hsv[:, :, 1]

        # Panel mask: dark grey, low saturation
        panel_mask = (val > 25) & (val < 95) & (sat < 45)
        row_coverage = panel_mask.sum(axis=1) / w

        # Find panel bottom: last row from top with >35% coverage (allow small gaps)
        panel_rows = np.where(row_coverage > 0.35)[0]
        if len(panel_rows) < 10:
            return "", ""
        panel_top = panel_rows[0]
        panel_bottom = panel_top
        for i in range(1, len(panel_rows)):
            if panel_rows[i] - panel_rows[i - 1] > 15:
                break
            panel_bottom = panel_rows[i]

        # Text mask: white pixels within panel
        text_mask = (val > 170) & (sat < 60)
        text_mask[panel_bottom:, :] = False  # ignore below panel

        # Find text row clusters
        text_row_counts = text_mask.sum(axis=1)
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

        if not clusters:
            return "", ""

        # Title = first cluster
        title_top, title_bottom = clusters[0]
        title_top = max(0, title_top - 2)
        title_bottom = min(h, title_bottom + 2)

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

        # Description: 3rd+ text cluster (skip category which is 2nd)
        desc_text = ""
        if len(clusters) >= 3 and (panel_bottom - panel_top) > 100:
            desc_top = max(0, clusters[2][0] - 2)
            desc_bottom = min(panel_bottom, clusters[-1][1] + 2)
            desc_col_mask = text_mask[desc_top:desc_bottom, :].sum(axis=0) > 0
            desc_cols = np.where(desc_col_mask)[0]
            if len(desc_cols) > 10:
                dl = max(0, desc_cols[0] - 5)
                dr = min(w, desc_cols[-1] + 5)
                desc_crop = crop[desc_top:desc_bottom, dl:dr]
                desc_gray = cv2.cvtColor(desc_crop, cv2.COLOR_BGR2GRAY)
                desc_inv = 255 - desc_gray
                _, desc_bin = cv2.threshold(desc_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                desc_padded = cv2.copyMakeBorder(desc_bin, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)
                desc_text = ocr_text(desc_padded, psm=7)
                desc_text = desc_text.strip().split('\n')[0].strip()

        return title_text, desc_text

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


    def _scan_grid_for_item(self, item_key: str):
        """Scan item grid by hovering each cell, reading tooltip via OCR.

        Returns a Box at the bottom-right-most match, or None.
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

        self.log_info(f"[grid_scan] targets={targets}, grid={self._GRID_COLS}x{self._GRID_ROWS}")
        if not _tess_mod._initialized:
            self.log_info("[grid_scan] WARNING: tesseract not initialized!")

        debug_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ok-ef" / "grid_scan_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

        for scroll_round in range(8):
            self.log_info(f"[grid_scan] --- round {scroll_round} ---")
            top_left_text = None

            for row in range(self._GRID_ROWS):
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

                # Phase 3: Read counts from clean frame (no tooltip overlay)
                row_counts = {}
                if clean_frame is not None:
                    for col, cx_px, cy_px, _, _ in row_captures:
                        cnt_x1 = max(0, cx_px - half_cw)
                        cnt_x2 = cx_px + half_cw
                        cnt_y1 = cy_px + half_ch - 55
                        cnt_y2 = min(h, cy_px + half_ch + 5)
                        count_str = ocr_text(clean_frame, box=(cnt_x1, cnt_y1, cnt_x2, cnt_y2), psm=7)
                        count_str = count_str.strip().split('\n')[0].strip()
                        row_counts[col] = self._parse_count(count_str)

                # Phase 4: OCR tooltips for item names
                for col, cx_px, cy_px, frame, (tx1, ty1, tx2, ty2) in row_captures:
                    # OCR tooltip via preprocessing (color-based text isolation)
                    tooltip_crop = frame[ty1:ty2, tx1:tx2]
                    title_text, desc_text = self._extract_tooltip_text(tooltip_crop)

                    try:
                        debug_frame = frame.copy()
                        cv2.rectangle(debug_frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 3)
                        cv2.circle(debug_frame, (cx_px, cy_px), 12, (0, 0, 255), -1)
                        cv2.imwrite(str(debug_dir / f"r{scroll_round}_cell_{row}_{col}.png"), debug_frame)
                        if tooltip_crop.size > 0:
                            cv2.imwrite(str(debug_dir / f"r{scroll_round}_tooltip_{row}_{col}.png"), tooltip_crop)
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
                                cv2.imwrite(str(debug_dir / f"r{scroll_round}_title_{row}_{col}.png"), title_debug)
                    except Exception:
                        pass

                    item_count = row_counts.get(col, 0)

                    if not title_text:
                        if col == 0 and scroll_round == 0:
                            self.log_info(f"[grid_scan] ({row},{col}) empty! diag={_ocr_diag}")
                        continue

                    full_text = title_text if not desc_text else f"{title_text} | {desc_text}"
                    count_label = f" x{item_count}" if item_count else ""
                    self.log_info(f"[grid_scan] ({row},{col}) -> '{full_text}'{count_label}")
                    seen_items.add(title_text.lower())

                    if row == 0 and col == 0 and top_left_text is None:
                        top_left_text = title_text

                    match_text = full_text.lower()
                    for target in targets:
                        if item_matches(match_text, target):
                            last_match = Box(cx_px - half_cw, cy_px - half_ch,
                                             int(col_width * w), int(row_height * h), name=title_text)
                            last_match_count = item_count
                            self.log_info(f"[grid_scan] MATCH '{target}' at ({row},{col}){count_label}")
                            break

            if last_match:
                self.log_info(f"[grid_scan] returning: '{last_match.name}' x{last_match_count}")
                return last_match

            if top_left_text and top_left_text == last_top_left_text:
                self.log_info(f"[grid_scan] end of list (top-left unchanged: '{top_left_text}')")
                break
            last_top_left_text = top_left_text

            # Scroll down ~4 rows
            scroll_x = int((self._GRID_LEFT + (self._GRID_RIGHT - self._GRID_LEFT) / 2) * w)
            scroll_y = int((self._GRID_TOP + (self._GRID_BOTTOM - self._GRID_TOP) / 2) * h)
            self._hover_absolute(scroll_x, scroll_y)
            self.sleep(0.2)
            self._scroll_precise(scroll_x, scroll_y, -230)
            self.sleep(1.2)
            self.next_frame()
            self.sleep(0.3)

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

            store_text = self.lang.WarehouseTransferTask.k_d661f6da
            self.log_info(f"[store] clicking Quick Stash")
            # Try Tesseract to verify Quick Stash button is visible
            self.next_frame()
            frame = self.executor.frame
            store_found = False
            if frame is not None:
                h, w = frame.shape[:2]
                store_box = (int(w * 0.78), int(h * 0.84), int(w * 0.97), int(h * 0.97))
                if ocr_match(frame, store_box, store_text):
                    self.log_info("[store] Tesseract confirmed Quick Stash visible")
                    store_found = True

            if not store_found:
                # Try framework OCR
                store_btn = self.wait_ocr(
                    box=self.box_of_screen(0.78, 0.84, 0.97, 0.97, name="onekey_store_area"),
                    match=store_text,
                    time_out=2,
                    raise_if_not_found=False,
                )
                if store_btn:
                    self.click(store_btn[0])
                    store_found = True

            if not store_found:
                self.log_info("[store] OCR missed, using positional click")
            # Click Quick Stash position regardless (confirmed visible or positional)
            self.click_relative(0.87, 0.90)
            self._maybe_click_confirm()
            max_times -= 1
            if max_times <= 0:
                break
            self.log_info(f"切回发货仓库={from_key}")
            self._switch_location(from_key)
        self.log_info("仓库转移任务完成")
