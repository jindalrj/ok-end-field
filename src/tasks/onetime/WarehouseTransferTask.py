import json
import os
from pathlib import Path

import cv2
import numpy as np
import win32api
import win32con
from qfluentwidgets import FluentIcon

from src.data.world_map import item_to_warehouse_dict
from src.data.zh_en import ITEM_WAREHOUSE_CATEGORY_EN_BY_ZH, ITEM_TRANSLATION_DICT
from src.core.BaseEfTask import BaseEfTask
from src.icons import Icons
from src.ocr import ocr_text, ocr_match, ensure_tesseract

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
                "物品": "蓝铁矿",
                "转移轮次": 10,
                # "最小保留数量": 1000,
            }
        )
        self.config_description.update(
            {
                "发货仓库": "从这个仓库拿货",
                "收货仓库": "转运到这个仓库",
                "物品": "选择要转移的物品",
                "转移轮次": "倒货的轮次",
                # "最小保留数量": "当识别到当前数量小于该值时停止任务并通知",
            }
        )
        _location_keys = list(_LOCATIONS.get("zh_CN", {}).keys())
        self.config_type["发货仓库"] = {"type": "drop_down", "options": _location_keys}
        self.config_type["收货仓库"] = {"type": "drop_down", "options": _location_keys}
        # Use all items that have template images available
        template_map = _load_item_template_map()
        available_items = [name for name in template_map if name in item_to_warehouse_dict]
        if not available_items:
            available_items = list(item_to_warehouse_dict.keys())
        self.config_type["物品"] = {
            "type": "drop_down",
            "options": available_items,
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
        category_en_name = ITEM_WAREHOUSE_CATEGORY_EN_BY_ZH.get(item_to_warehouse_dict.get(item_name, ""), "")
        if not category_en_name:
            raise ValueError(f"物品 {item_name} 无法找到分类，无法定位图标")
        result = self.find_feature(feature=f"{category_en_name}_icon")
        if not result:
            self.log_info(f"物品 {item_name} 无法找到分类图标,可能已经进入该分类页")
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
            from src.ocr import ocr_frame
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

    # Grid layout constants (relative to full game frame)
    # Measured from runtime screenshots: items start at ~(0.065, 0.15)
    # Grid is 8 columns wide, 4 visible rows before needing to scroll
    _GRID_LEFT = 0.065
    _GRID_TOP = 0.155
    _GRID_RIGHT = 0.535
    _GRID_BOTTOM = 0.62
    _GRID_COLS = 8
    _GRID_ROWS = 4  # visible rows before scrolling

    def _hover_absolute(self, px_x: int, px_y: int):
        """Move the real cursor AND send WM_MOUSEMOVE to trigger game tooltip.

        Uses SetCursorPos for physical cursor + PostMessage WM_MOUSEMOVE
        so the game processes the hover event (needed for tooltip popup).
        """
        import win32gui
        # Activate window and move real cursor
        interaction = self.executor.interaction
        interaction.try_activate()
        abs_x, abs_y = interaction.capture.get_abs_cords(px_x, px_y)
        win32api.SetCursorPos((abs_x, abs_y))
        # Also send WM_MOUSEMOVE to the game window so it processes the hover
        lParam = win32api.MAKELONG(px_x, px_y)
        win32gui.PostMessage(interaction.hwnd, win32con.WM_MOUSEMOVE, 0, lParam)

    def _scan_grid_for_item(self, item_key_zh: str):
        """
        Scan the item grid by hovering each cell and reading the tooltip via OCR.

        Moves left-to-right, top-to-bottom. After scanning all visible rows,
        scrolls down and repeats. Stops when target is found or bottom-right
        cell has been seen twice (indicating no more items to scroll).

        Args:
            item_key_zh: Chinese item name from config (used for matching)

        Returns a Box-like object at the item center, or None.
        """
        from ok.feature.Box import Box

        # Build match targets: Chinese name + English code name (spaces/underscores normalized)
        targets = [item_key_zh.lower()]
        en_code = ITEM_TRANSLATION_DICT.get(item_key_zh, "")
        if en_code:
            # "dense_source_ore_powder" -> match "dense" "source" "ore" "powder" as substrings
            targets.append(en_code.replace("_", " ").lower())

        w, h = self.width, self.height
        col_width = (self._GRID_RIGHT - self._GRID_LEFT) / self._GRID_COLS
        row_height = (self._GRID_BOTTOM - self._GRID_TOP) / self._GRID_ROWS

        seen_items = set()
        last_bottom_right_text = None
        MAX_SCROLL_ROUNDS = 8

        self.log_info(f"[grid_scan] targets={targets}, grid={self._GRID_COLS}x{self._GRID_ROWS}, "
                      f"cell_size=({col_width:.3f}, {row_height:.3f})")

        # Debug screenshot directory
        debug_dir = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "ok-ef" / "grid_scan_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        self.log_info(f"[grid_scan] debug screenshots -> {debug_dir}")

        for scroll_round in range(MAX_SCROLL_ROUNDS):
            self.log_info(f"[grid_scan] --- scroll round {scroll_round} ---")
            bottom_right_text = None

            for row in range(self._GRID_ROWS):
                for col in range(self._GRID_COLS):
                    # Center of this grid cell (relative coords)
                    cx_rel = self._GRID_LEFT + col_width * (col + 0.5)
                    cy_rel = self._GRID_TOP + row_height * (row + 0.5)
                    cx_px = int(cx_rel * w)
                    cy_px = int(cy_rel * h)

                    # Hover using real cursor (required for tooltip to appear)
                    self._hover_absolute(cx_px, cy_px)
                    self.sleep(0.6)  # tooltip takes ~0.5s to appear

                    # Capture frame and OCR the tooltip region
                    self.next_frame()
                    frame = self.executor.frame
                    if frame is None:
                        self.log_info(f"[grid_scan] ({row},{col}) frame is None!")
                        continue

                    # Tooltip region: the tooltip appears as a small card overlapping
                    # the top portion of the hovered item. Scan from above the cell
                    # center down to just below it, wide enough to catch offset tooltips.
                    tooltip_y1 = max(0, int((cy_rel - row_height * 0.7) * h))
                    tooltip_y2 = int((cy_rel + row_height * 0.1) * h)
                    tooltip_x1 = max(0, int((cx_rel - col_width * 1.5) * w))
                    tooltip_x2 = min(w, int((cx_rel + col_width * 1.5) * w))

                    # Save debug screenshots for first 8 cells (one full row)
                    if scroll_round == 0 and row == 0:
                        try:
                            debug_frame = frame.copy()
                            cv2.rectangle(debug_frame, (tooltip_x1, tooltip_y1), (tooltip_x2, tooltip_y2), (0, 255, 0), 2)
                            cv2.circle(debug_frame, (cx_px, cy_px), 10, (0, 0, 255), -1)
                            cv2.imwrite(str(debug_dir / f"r{scroll_round}_cell_{row}_{col}.png"), debug_frame)
                            # Also save just the tooltip crop
                            crop = frame[tooltip_y1:tooltip_y2, tooltip_x1:tooltip_x2]
                            if crop.size > 0:
                                cv2.imwrite(str(debug_dir / f"r{scroll_round}_tooltip_{row}_{col}.png"), crop)
                        except Exception as e:
                            self.log_info(f"[grid_scan] screenshot save failed: {e}")

                    text = ocr_text(frame, box=(tooltip_x1, tooltip_y1, tooltip_x2, tooltip_y2), psm=6)
                    text = text.strip()
                    if not text and row == 0 and col == 0 and scroll_round == 0:
                        # First cell empty = likely Tesseract not initialized
                        from src.ocr.tesseract_ocr import _initialized
                        self.log_info(f"[grid_scan] first cell empty! tesseract _initialized={_initialized}, "
                                      f"tooltip_box=({tooltip_x1},{tooltip_y1},{tooltip_x2},{tooltip_y2}), "
                                      f"frame_shape={frame.shape}")

                    if text:
                        self.log_info(f"[grid_scan] ({row},{col}) pos=({cx_rel:.3f},{cy_rel:.3f}) -> '{text}'")
                        seen_items.add(text.lower())

                        # Track bottom-right cell text for scroll termination
                        if row == self._GRID_ROWS - 1 and col == self._GRID_COLS - 1:
                            bottom_right_text = text

                        # Check if this is our target (substring match)
                        text_lower = text.lower()
                        for target in targets:
                            if target in text_lower or text_lower in target:
                                self.log_info(f"[grid_scan] FOUND target '{target}' in '{text}' at ({row},{col})")
                                return Box(cx_px - int(col_width * w / 2), cy_px - int(row_height * h / 2),
                                           int(col_width * w), int(row_height * h), name=text)
                    else:
                        self.log_info(f"[grid_scan] ({row},{col}) empty")

            # Check if we've hit the bottom (same content as last scroll)
            if bottom_right_text and bottom_right_text == last_bottom_right_text:
                self.log_info(f"[grid_scan] bottom-right unchanged ('{bottom_right_text}'), no more items")
                break
            last_bottom_right_text = bottom_right_text

            # Scroll down to reveal more items
            scroll_x = int((self._GRID_LEFT + (self._GRID_RIGHT - self._GRID_LEFT) / 2) * w)
            scroll_y = int((self._GRID_TOP + (self._GRID_BOTTOM - self._GRID_TOP) / 2) * h)
            self.move(scroll_x, scroll_y)
            self.scroll(scroll_x, scroll_y, -3)
            self.sleep(0.6)

        self.log_info(f"[grid_scan] targets {targets} NOT found. Seen: {sorted(seen_items)}")
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
            ensure_tesseract()
            self.log_info("[run] Tesseract OCR initialized")
        except Exception as e:
            self.log_info(f"[run] Tesseract not available ({e}), using positional clicks only")

        from_key = str(self.config.get("发货仓库", "wuling")).strip()
        to_key = str(self.config.get("收货仓库", "valley4")).strip()
        if from_key == to_key:
            raise RuntimeError("发货仓库与收货仓库不能相同")

        item_key = str(self.config.get("物品", "")).strip()
        if not item_key:
            raise RuntimeError("未选择物品")
        max_times = int(self.config.get("转移轮次", 10))
        locale = self._get_locale()
        self.log_info(f"[run] from={from_key}, to={to_key}, item={item_key}, rounds={max_times}, locale={locale}, resolution={self.width}x{self.height}")
        self.press_key("b")
        self.sleep(2.0)
        # Wait for depot UI to appear (try OCR, but don't block on it)
        self.wait_until(
            lambda: self._detect_current_location() is not None,
            time_out=5,
            raise_if_not_found=False,
        )
        while True:
            current = self._detect_current_location()
            if current != from_key:
                self.log_info(f"当前仓库={current}，切换到发货仓库={from_key}")
                self._switch_location(from_key)
            self._to_one_type_page(item_key)
            self.log_info(f"处理物品: {item_key}")

            # Find item by grid-scanning with hover + OCR
            # Match against both Chinese name and English tooltip name
            icon = self._scan_grid_for_item(item_key)

            if not icon:
                raise RuntimeError(f"未找到物品图标（网格扫描后仍失败）：{item_key}")
            self._ctrl_click(icon)
            self.sleep(0.35)

            self.log_info(f"切换到收货仓库={to_key}")
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
