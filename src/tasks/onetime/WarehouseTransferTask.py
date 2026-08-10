import json
from pathlib import Path

import cv2
import win32api
import win32con
from qfluentwidgets import FluentIcon

from src.data.world_map import item_to_warehouse_dict
from src.data.zh_en import ITEM_WAREHOUSE_CATEGORY_EN_BY_ZH, ITEM_TRANSLATION_DICT
from src.core.BaseEfTask import BaseEfTask
from src.icons import Icons

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
        """Load the template image for an item, returning a cv2 ndarray or None."""
        if item_name in self._template_cache:
            return self._template_cache[item_name]
        template_map = _load_item_template_map()
        template_filename = template_map.get(item_name)
        if not template_filename:
            self._template_cache[item_name] = None
            return None
        template_path = _ITEM_TEMPLATE_DIR / f"{template_filename}.png"
        if not template_path.exists():
            self._template_cache[item_name] = None
            return None
        img = cv2.imread(str(template_path), cv2.IMREAD_COLOR)
        self._template_cache[item_name] = img
        return img

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
        """Try OCR detection, but return None gracefully if OCR doesn't work."""
        self.next_frame()
        detect_box = self.box_of_screen(0.02, 0.03, 0.22, 0.10, name="current_location_area")
        boxes = self.ocr(box=detect_box, threshold=0.1)
        locale = self._get_locale()
        detect_map = _LOCATION_DETECT.get(locale, _LOCATION_DETECT["zh_CN"])

        all_texts = []
        for box in boxes or []:
            text = str(getattr(box, "name", "")).strip()
            if text:
                all_texts.append(text)
        combined_text = " ".join(all_texts)
        if all_texts:
            self.log_info(f"[detect_location] OCR texts={all_texts}")

        # Per-box matching
        for box in boxes or []:
            name = str(getattr(box, "name", "")).strip()
            for loc_key, pattern in detect_map.items():
                if isinstance(pattern, tuple):
                    if all(part in name for part in pattern):
                        return loc_key
                elif pattern in name:
                    return loc_key

        # Combined text matching
        for loc_key, pattern in detect_map.items():
            if isinstance(pattern, tuple):
                if all(part in combined_text for part in pattern):
                    return loc_key
            elif pattern in combined_text:
                return loc_key
        return None

    def _maybe_click_confirm(self):
        """Click Confirm button if it appears (positional click with OCR fallback)."""
        # Try OCR first
        confirm_text = self.lang.WarehouseTransferTask.k_b56d9ac6
        hits = self.wait_ocr(
            box=self.box_of_screen(0.78, 0.84, 0.97, 0.97, name="confirm_btn_area"),
            match=confirm_text,
            time_out=2,
            raise_if_not_found=False,
        )
        if hits:
            self.log_info(f"[confirm] found via OCR, clicking")
            self.click(hits[0])
            self.sleep(0.5)
            return True
        # Positional fallback: click where Confirm button appears
        # From screenshots: yellow Confirm button at ~(0.88, 0.91)
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
        # From screenshots: center ~(0.26, 0.06)
        switch_text = self.lang.WarehouseTransferTask.k_3cb6baa6
        self.log_info(f"[switch] clicking Switch Depot button, locale={locale}")
        btn = self.wait_ocr(
            box=self.box_of_screen(0.20, 0.03, 0.35, 0.10, name="switch_btn_area"),
            match=switch_text,
            time_out=3,
            raise_if_not_found=False,
        )
        if btn:
            self.click(btn[0])
        else:
            self.log_info("[switch] OCR missed button, using positional click")
            self.click_relative(0.26, 0.06)
        self.sleep(1.0)

        # Step 2: Click the target depot option in the modal
        # From screenshots: Wuling ~(0.35, 0.14), Valley IV ~(0.35, 0.18)
        _OPTION_POSITIONS = {"wuling": 0.14, "valley4": 0.18}
        target_text = loc_names[target_key]
        self.log_info(f"[switch] selecting depot '{target_text}'")
        option = self.wait_ocr(
            box=self.box_of_screen(0.15, 0.08, 0.55, 0.30, name="switch_menu"),
            match=target_text,
            time_out=3,
            raise_if_not_found=False,
        )
        if option:
            self.click(option[0])
        else:
            option_y = _OPTION_POSITIONS.get(target_key, 0.16)
            self.log_info(f"[switch] OCR missed option, using positional click y={option_y}")
            self.click_relative(0.35, option_y)
        self.sleep(0.5)

        # Step 3: Click Confirm if it appears
        self._maybe_click_confirm()

        # Step 4: Wait for connection (try OCR, fall back to fixed wait)
        connected_text = self.lang.WarehouseTransferTask.k_65fe35c4
        self.log_info(f"[switch] waiting for connection...")
        connected = False
        for i in range(20):
            self.next_frame()
            hits = self.ocr(box=self.box.bottom_right, match=connected_text, threshold=0.1)
            if hits:
                self.log_info(f"[switch] connected detected after {i} polls")
                connected = True
                break
            self.sleep(0.5)
        if not connected:
            # OCR may not work for English — just wait a fixed time
            self.log_info("[switch] OCR didn't detect 'Connected', waiting fixed 5s")
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
        search_box = self.box_of_screen(0.12, 0.30, 0.55, 0.68)
        while True:
            current = self._detect_current_location()
            if current != from_key:
                self.log_info(f"当前仓库={current}，切换到发货仓库={from_key}")
                self._switch_location(from_key)
            self._to_one_type_page(item_key)
            cx = int(self.width / 3)
            cy = int(self.height * 0.5)
            self.log_info(f"处理物品: {item_key}")

            ROUND = 5
            icon = None
            item_template = self._load_item_template(item_key)
            item_key_en = ITEM_TRANSLATION_DICT.get(item_key, "")

            for round_idx in range(ROUND + 1):
                if item_template is not None:
                    # Use direct template matching (resolution-independent via target_height)
                    icon = self.find_one(
                        feature="item_template_match",
                        template=item_template,
                        box=search_box,
                        threshold=0.7,
                        target_height=180,
                    )
                elif item_key_en:
                    # Fallback: use COCO feature if no template image available
                    icon = self.find_one(feature=item_key_en, box=search_box, threshold=0.8)
                else:
                    raise RuntimeError(f"No template or feature found for item: {item_key}")
                if icon:
                    break
                if round_idx == ROUND:
                    break
                self.move(cx, cy)
                self.scroll(cx, cy, -2)
                self.sleep(0.5)

            if not icon:
                raise RuntimeError(f"未找到物品图标（滚动{ROUND}轮后仍失败）：{item_key}")
            self._ctrl_click(icon)
            self.sleep(0.35)
            if item_template is not None:
                icon_after = self.find_feature(
                    feature="item_template_match",
                    template=item_template,
                    box=search_box,
                    threshold=0.7,
                    target_height=180,
                )
            elif item_key_en:
                icon_after = self.find_feature(feature=item_key_en, box=search_box, threshold=0.8)
            else:
                icon_after = None
            if not icon_after:
                self.log_info(f"物品图标已消失（可能已倒完）：{item_key}")
                # count_after = self._read_count_near_icon(icon_after)
                # if count_before is not None and count_after is not None:
                #     self.log_debug(f"物品数量(后): {count_after}")
                #     if count_after >= count_before:
                #         raise RuntimeError(f"点击后数量未减少：{item_key} 前={count_before} 后={count_after}")

            self.log_info(f"切换到收货仓库={to_key}")
            self._switch_location(to_key)

            store_text = self.lang.WarehouseTransferTask.k_d661f6da
            self.log_info(f"[store] clicking Quick Stash")
            store_btn = self.wait_ocr(
                box=self.box_of_screen(0.78, 0.84, 0.97, 0.97, name="onekey_store_area"),
                match=store_text,
                time_out=3,
                raise_if_not_found=False,
            )
            if store_btn:
                self.click(store_btn[0])
            else:
                # Positional fallback: Quick Stash at ~(0.87, 0.90)
                self.log_info("[store] OCR missed, using positional click")
                self.click_relative(0.87, 0.90)
            self._maybe_click_confirm()
            max_times -= 1
            if max_times <= 0:
                break
            self.log_info(f"切回发货仓库={from_key}")
            self._switch_location(from_key)
        self.log_info("仓库转移任务完成")
