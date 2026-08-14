"""
Multi-item cross-depot transfer with automatic round planning.

Two item groups, each with its own transfer direction (valley4 <-> wuling).
The task first scans both depots to inventory the selected items, then
filters transfers by capacity rules:
  - destination must have > 5K free capacity for the item
    (per-item depot limits: Valley IV 80K, Wuling 90K)
  - source must hold > 2K of the item
Eligible transfers are then executed one by one, round by round, until a
limit is reached - no manual round counting needed.

Reuses the verified scan/switch/deposit machinery from WarehouseTransferTask.
"""

from qfluentwidgets import FluentIcon

from src.data.world_map import item_to_warehouse_dict
from src.data.zh_en import ITEM_GAME_ENGLISH
from src.icons import Icons
from src.ocr import ensure_tesseract
from src.tasks.onetime.item_matcher import match_score as item_match_score
from src.tasks.onetime.WarehouseTransferTask import WarehouseTransferTask, _MATCH_VOCAB

_DIR_V2W = "valley4 -> wuling"
_DIR_W2V = "wuling -> valley4"

# Per-item depot capacity: Valley IV caps at 80K, Wuling at 90K.
_DEPOT_LIMITS = {"valley4": 80000, "wuling": 90000}
_DEST_HEADROOM = 5000   # destination needs > 5K free capacity
_SOURCE_MIN = 2000      # source must keep / have > 2K
_MAX_ROUNDS_PER_ITEM = 30  # safety cap per item


class MultiItemWarehouseTransferTask(WarehouseTransferTask):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = "多物品仓库转移"
        self.icon = Icons.ItemTransfer
        self.group_name = "仓储管理"
        self.group_icon = FluentIcon.FOLDER
        self.description = "勾选多个物品自动倒仓：先盘点两侧仓库库存，再按容量规则逐一转移"

        # Drop the single-item options inherited from WarehouseTransferTask
        for k in ("发货仓库", "收货仓库", "物品", "转移轮次"):
            self.default_config.pop(k, None)
            self.config_description.pop(k, None)
            self.config_type.pop(k, None)

        _en_items = sorted(ITEM_GAME_ENGLISH.values())
        _dir_options = [_DIR_V2W, _DIR_W2V]
        self.default_config.update({
            "转移方向1": _DIR_V2W,
            "转移物品1": [],
            "转移方向2": _DIR_W2V,
            "转移物品2": [],
        })
        self.config_description.update({
            "转移方向1": "Transfer direction for item group 1",
            "转移物品1": "Items to transfer in direction 1",
            "转移方向2": "Transfer direction for item group 2",
            "转移物品2": "Items to transfer in direction 2",
        })
        self.config_type.update({
            "转移方向1": {"type": "drop_down", "options": _dir_options},
            "转移物品1": {"type": "multi_selection", "options": _en_items},
            "转移方向2": {"type": "drop_down", "options": _dir_options},
            "转移物品2": {"type": "multi_selection", "options": _en_items},
        })

    # ---------- config parsing ----------

    @staticmethod
    def _parse_direction(value: str) -> tuple[str, str]:
        parts = [p.strip() for p in str(value).split("->")]
        if (len(parts) == 2 and parts[0] in _DEPOT_LIMITS
                and parts[1] in _DEPOT_LIMITS and parts[0] != parts[1]):
            return parts[0], parts[1]
        raise RuntimeError(f"Invalid transfer direction: {value!r}")

    def _parse_transfers(self) -> list[dict]:
        """Build the transfer list from both config groups.

        Duplicates within the same direction are collapsed; the same item
        in BOTH directions is a config error.
        """
        transfers = []
        direction_by_item: dict[str, tuple[str, str]] = {}
        for dir_key, items_key in (("转移方向1", "转移物品1"), ("转移方向2", "转移物品2")):
            from_key, to_key = self._parse_direction(self.config.get(dir_key, _DIR_V2W))
            for item in self.config.get(items_key) or []:
                item = str(item).strip()
                if not item:
                    continue
                prev = direction_by_item.get(item)
                if prev == (from_key, to_key):
                    continue
                if prev is not None:
                    raise RuntimeError(f"'{item}' is selected in both directions")
                direction_by_item[item] = (from_key, to_key)
                transfers.append({"item": item, "from": from_key, "to": to_key})
        return transfers

    # ---------- inventory ----------

    def _item_category(self, item: str) -> str:
        zh = item
        if item not in item_to_warehouse_dict:
            en_to_zh = {v.lower(): k for k, v in ITEM_GAME_ENGLISH.items()}
            zh = en_to_zh.get(item.lower(), item)
        return item_to_warehouse_dict.get(zh, "")

    def _scroll_grid_to_top(self):
        """Scroll the item grid back to the top (3 page-up scrolls)."""
        w, h = self.width, self.height
        x = int((self._GRID_LEFT + (self._GRID_RIGHT - self._GRID_LEFT) / 2) * w)
        y = int((self._GRID_TOP + (self._GRID_BOTTOM - self._GRID_TOP) / 2) * h)
        self._hover_absolute(x, y)
        self.sleep(0.2)
        for _ in range(3):
            self._scroll_precise(x, y, 230)
            self.sleep(0.4)
        self.next_frame()

    def _scan_counts_for_items(self, items: list[str], depot: str) -> dict[str, int]:
        """Scan the current category page and total the counts per item.

        Consecutive pages can overlap when the final scroll clamps at the
        list end (scroll moves a full 4-row page normally). Overlapping
        leading rows - identical (text, count) signature to the previous
        page's trailing rows - are skipped to avoid double counting.
        """
        targets_by_item = {it: self._build_match_targets(it) for it in items}
        counts = {it: 0 for it in items}
        prev_sigs = None

        for scroll_round in range(12):
            self.log_info(f"[inventory] {depot} --- page {scroll_round} ---")
            page_rows = []  # list of (row_signature, row_cells)
            for row in range(self._GRID_ROWS):
                cells = self._scan_row_cells(row, scroll_round, debug_prefix=f"inv_{depot}_")
                row_cells = []
                for cell in cells:
                    title, desc, count = cell["title"], cell["desc"], cell["count"]
                    full_text = ""
                    if title:
                        full_text = (title if not desc else f"{title} | {desc}").lower()
                        self.log_info(f"[inventory] ({row},{cell['col']}) -> '{full_text}' x{count}")
                    row_cells.append((full_text, count))
                page_rows.append((tuple(row_cells), row_cells))

            sigs = [sig for sig, _ in page_rows]
            overlap = 0
            if prev_sigs is not None:
                for k in range(min(len(prev_sigs), len(sigs)), 0, -1):
                    if prev_sigs[-k:] == sigs[:k]:
                        overlap = k
                        break
                if overlap:
                    self.log_info(f"[inventory] skipping {overlap} overlapping row(s)")

            for _, row_cells in page_rows[overlap:]:
                for full_text, count in row_cells:
                    if not full_text:
                        continue
                    best_item, best_score = None, 0.0
                    for it, targets in targets_by_item.items():
                        s = max(item_match_score(full_text, t, _MATCH_VOCAB) for t in targets)
                        if s > best_score:
                            best_item, best_score = it, s
                    if best_item:
                        if count <= 0:
                            self.log_info(f"[inventory] WARNING: '{full_text}' matched "
                                          f"'{best_item}' but count OCR is 0 - total may be low")
                        counts[best_item] += count

            if prev_sigs is not None and overlap == len(sigs):
                self.log_info("[inventory] page identical to previous - end of list")
                break
            if all(not ft for _, row_cells in page_rows for ft, _ in row_cells):
                self.log_info("[inventory] empty page - end of list")
                break
            prev_sigs = sigs
            self._scroll_grid_page()

        return counts

    def _inventory_depot(self, depot: str, items: list[str]) -> dict[str, int]:
        """Inventory the given items at the current depot, one category page at a time."""
        by_cat: dict[str, list[str]] = {}
        for it in items:
            by_cat.setdefault(self._item_category(it), []).append(it)
        counts: dict[str, int] = {}
        for cat, cat_items in sorted(by_cat.items()):
            self.log_info(f"[inventory] {depot}: category '{cat}' items={cat_items}")
            self._to_one_type_page(cat_items[0])
            self._scroll_grid_to_top()
            counts.update(self._scan_counts_for_items(cat_items, depot))
        return counts

    # ---------- transfer ----------

    def _transfer_item(self, item: str, from_key: str, to_key: str,
                       src_count: int, dst_count: int) -> int:
        """Transfer one item round by round until a capacity rule stops it.

        Counts are tracked incrementally from the withdrawn cell counts,
        so no re-inventory is needed between rounds.
        """
        limit = _DEPOT_LIMITS[to_key]
        moved_total = 0
        for _ in range(_MAX_ROUNDS_PER_ITEM):
            if src_count <= _SOURCE_MIN:
                self.log_info(f"[transfer] '{item}': source down to ~{src_count}, done")
                break
            if dst_count >= limit - _DEST_HEADROOM:
                self.log_info(f"[transfer] '{item}': destination up to ~{dst_count} "
                              f"(limit {limit}), done")
                break
            if self._detect_current_location() != from_key:
                self._switch_location(from_key)
            self._to_one_type_page(item)
            icon = self._scan_grid_for_item(item)
            if not icon:
                self.log_info(f"[transfer] '{item}': no longer found in source, done")
                break
            moved = self._last_scan_count
            self._ctrl_click(icon)
            self.sleep(0.35)
            self._switch_location(to_key)
            self._do_deposit()
            if moved <= 0:
                self.log_info(f"[transfer] '{item}': withdrawn count unknown, "
                              f"stopping this item for safety")
                break
            src_count -= moved
            dst_count += moved
            moved_total += moved
            self.log_info(f"[transfer] '{item}': moved {moved} this round "
                          f"(total {moved_total}), source~{src_count}, dest~{dst_count}")
        return moved_total

    def run(self):
        self.ensure_main()
        try:
            diag = ensure_tesseract()
            self.log_info(f"[run] Tesseract: {diag}")
        except Exception as e:
            self.log_info(f"[run] Tesseract FAILED: {e}")

        transfers = self._parse_transfers()
        if not transfers:
            raise RuntimeError("No items selected")
        self.log_info("[run] requested: " + ", ".join(
            f"{t['item']} ({t['from']}->{t['to']})" for t in transfers))

        # Each transfer needs counts at BOTH its source and destination
        items_by_depot: dict[str, set] = {}
        for t in transfers:
            items_by_depot.setdefault(t["from"], set()).add(t["item"])
            items_by_depot.setdefault(t["to"], set()).add(t["item"])

        self.press_key("b")
        self.sleep(2.0)
        self.wait_until(
            lambda: self._detect_current_location() is not None,
            time_out=5,
            raise_if_not_found=False,
        )

        # Phase 1: inventory (current depot first to save one switch)
        current = self._detect_current_location()
        depot_order = sorted(items_by_depot.keys(), key=lambda d: d != current)
        inventory: dict[tuple[str, str], int] = {}
        for depot in depot_order:
            if self._detect_current_location() != depot:
                self._switch_location(depot)
            counts = self._inventory_depot(depot, sorted(items_by_depot[depot]))
            for it, c in counts.items():
                inventory[(depot, it)] = c
                self.log_info(f"[inventory] {depot}: '{it}' = {c}")

        # Phase 2: eligibility filter
        eligible = []
        for t in transfers:
            item, from_key, to_key = t["item"], t["from"], t["to"]
            src_c = inventory.get((from_key, item), 0)
            dst_c = inventory.get((to_key, item), 0)
            limit = _DEPOT_LIMITS[to_key]
            if src_c <= _SOURCE_MIN:
                self.log_info(f"[plan] SKIP '{item}': source {from_key} has "
                              f"{src_c} (needs > {_SOURCE_MIN})")
                continue
            if dst_c >= limit - _DEST_HEADROOM:
                self.log_info(f"[plan] SKIP '{item}': destination {to_key} has "
                              f"{dst_c} (needs < {limit - _DEST_HEADROOM})")
                continue
            self.log_info(f"[plan] TRANSFER '{item}': {from_key}({src_c}) -> {to_key}({dst_c})")
            eligible.append((item, from_key, to_key, src_c, dst_c))

        if not eligible:
            self.log_info("[run] no eligible transfers after inventory", notify=True)
            return

        # Phase 3: execute transfers one by one
        summary = []
        for item, from_key, to_key, src_c, dst_c in eligible:
            self.log_info(f"[run] === '{item}': {from_key} -> {to_key} ===")
            moved = self._transfer_item(item, from_key, to_key, src_c, dst_c)
            summary.append(f"{item}: {moved}")
        self.log_info("[run] multi-item transfer complete: " + "; ".join(summary), notify=True)
