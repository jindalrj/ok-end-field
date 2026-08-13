"""
Item matching for OCR-scanned tooltip text against target item names.

Decomposes items by type (filled container, battery, graded consumable, simple)
and applies appropriate matching strategy for each.
"""

import re
from difflib import SequenceMatcher
from enum import Enum, auto


class ItemType(Enum):
    FILLED_CONTAINER = auto()  # "Hetonite Bottle (Clean Water)"
    BATTERY = auto()           # "LC Valley Battery", "SC Wuling Battery"
    GRADED = auto()            # "Buck Capsule [A]", "Yazhen Syringe [C]"
    SIMPLE = auto()            # "Ferrium Ore", "Heavy Xiranite"


_PAREN_RE = re.compile(r'^(.+?)\s*\((.+?)\)$')
_BRACKET_RE = re.compile(r'^(.+?)\s*\[(.+?)\]$')
_PREFIX_RE = re.compile(r'^(LC|SC|HC)\s+', re.IGNORECASE)
_FILLED_WITH_RE = re.compile(r'^filled with\s+', re.IGNORECASE)
_OCR_CLEAN_RE = re.compile(r'[^a-z0-9\s\'\-\[\]|]')


def classify_target(target: str) -> tuple[ItemType, dict]:
    """Classify an item target string and extract its components.

    Returns (item_type, components_dict) where components vary by type:
      FILLED_CONTAINER: {"base": str, "fill": str}
      BATTERY:          {"prefix": str, "rest": str}
      GRADED:           {"base": str, "grade": str}
      SIMPLE:           {"name": str}
    """
    m = _PAREN_RE.match(target)
    if m:
        return ItemType.FILLED_CONTAINER, {"base": m.group(1).strip(), "fill": m.group(2).strip()}

    m = _BRACKET_RE.match(target)
    if m:
        return ItemType.GRADED, {"base": m.group(1).strip(), "grade": m.group(2).strip()}

    m = _PREFIX_RE.match(target)
    if m:
        return ItemType.BATTERY, {"prefix": m.group(1).lower(), "rest": target.strip()}

    return ItemType.SIMPLE, {"name": target}


def parse_ocr_text(ocr_text: str) -> tuple[str, str]:
    """Parse OCR output into (title, description) from band-split format.

    Input is lowercase, potentially containing "|" separator from band OCR:
      "hetonite bottle | filled with clean water"
    Returns:
      ("hetonite bottle", "clean water")  -- "Filled with" prefix stripped
    """
    clean = _OCR_CLEAN_RE.sub('', ocr_text).strip()
    parts = [p.strip() for p in clean.split('|')]
    title = parts[0]
    desc = parts[1] if len(parts) > 1 else ""
    desc = _FILLED_WITH_RE.sub('', desc).strip()
    return title, desc


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def match_filled_container(title: str, desc: str, base: str, fill: str) -> bool:
    """Match a filled container: title must match base, desc must match fill."""
    if _ratio(title, base) < 0.82:
        return False
    if not desc:
        return False
    return _ratio(desc, fill) >= 0.80


def match_battery(title: str, prefix: str, full_target: str) -> bool:
    """Match a battery: require exact 2-char prefix, then fuzzy on full name."""
    title_prefix_match = re.match(r'^([a-z]{2})\s+', title)
    if title_prefix_match:
        if title_prefix_match.group(1) != prefix:
            return False
    return _ratio(title, full_target) >= 0.85


def match_graded(title: str, base: str, grade: str) -> bool:
    """Match a graded item: require exact bracket content, fuzzy on base name."""
    title_bracket = re.search(r'\[(.+?)\]', title)
    if title_bracket:
        if title_bracket.group(1).strip().lower() != grade.lower():
            return False
        title_base = re.sub(r'\s*\[.+?\]', '', title).strip()
        return _ratio(title_base, base) >= 0.82
    # No bracket found in OCR - require high overall similarity
    full_target = f"{base} [{grade}]"
    return _ratio(title, full_target) >= 0.85


def match_simple(title: str, target_name: str) -> bool:
    """Match a simple item: substring or fuzzy (ratio >= 0.82).

    Requires both title and target to be at least 3 chars to avoid
    garbage matches like 'w' or '—' matching real item names.
    """
    if len(title) < 3 or len(target_name) < 3:
        return False
    if target_name in title or title in target_name:
        return True
    return _ratio(title, target_name) >= 0.82


def _is_garbage(text: str) -> bool:
    """Return True if text is likely OCR garbage (symbols, single chars, etc.)."""
    if len(text) < 3:
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return True
    return False


def matches(ocr_text: str, target: str) -> bool:
    """Check if OCR text matches a target item name.

    Args:
        ocr_text: Lowercase OCR output, potentially "title | description" format
        target: Lowercase target item name from ITEM_GAME_ENGLISH

    Returns True if the OCR text identifies the target item.
    """
    if not ocr_text or not target:
        return False

    title, desc = parse_ocr_text(ocr_text)

    # Reject garbage OCR output before attempting match
    if _is_garbage(title):
        return False

    item_type, components = classify_target(target)

    if item_type == ItemType.FILLED_CONTAINER:
        return match_filled_container(title, desc, components["base"], components["fill"])

    if item_type == ItemType.BATTERY:
        return match_battery(title, components["prefix"], components["rest"])

    if item_type == ItemType.GRADED:
        return match_graded(title, components["base"], components["grade"])

    return match_simple(title, components["name"])
