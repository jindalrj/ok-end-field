"""
Item matching for OCR-scanned tooltip text against target item names.

Decomposes items by type (filled container, battery, graded consumable, simple)
and applies appropriate matching strategy for each.

`match_score` returns a similarity score (0.0 = no match) so callers can pick
the BEST-scoring cell instead of the first/last cell above a threshold -
'Pyrrolite' must not beat 'Pyrrolite Component', and a bottle of
'Liquid Xiranite' must not beat one of 'Liquid Heavy Xiranite'.
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
# Word that is a roman numeral or number: grade suffixes like 'I'/'II'/'III'
# must match exactly ('aerospace material i' vs 'ii' differ by one char but
# are DIFFERENT items; fuzzy ratio 0.67 would pass a short-word threshold).
_NUMERAL_RE = re.compile(r'^(?:[ivx]+|\d+)$')


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


def _word_match_score(text: str, target: str, vocab: frozenset | None = None) -> float:
    """Score how well `text` names exactly the item `target`.

    Every target word must fuzzy-match a distinct word in text (numeral
    words like 'i'/'ii' must match exactly). Leftover text words that look
    like real item-name words mean the text names a DIFFERENT, longer-named
    item ('pyrrolite' != 'pyrrolite component', 'dense carbon powder' !=
    'carbon powder', 'ferrium ore' != 'ferrium') -> 0.0. A leftover word is
    "real" if it has >=4 alphabetic chars or appears in `vocab` (the set of
    all words used in known item names - distinguishes 'ore' from OCR junk
    like 'joo'). Short non-vocab junk ('e<', 'yo.', 'qj') is ignored.

    Returns 0.0 on no match, else mean per-word ratio in (0, 1].
    """
    t_words = target.split()
    x_words = text.split()
    if not t_words or not x_words:
        return 0.0

    used = set()
    total = 0.0
    for tw in t_words:
        if _NUMERAL_RE.match(tw):
            hit = next((j for j, xw in enumerate(x_words)
                        if j not in used and xw == tw), -1)
            if hit < 0:
                return 0.0
            used.add(hit)
            total += 1.0
            continue
        best, best_j = 0.0, -1
        for j, xw in enumerate(x_words):
            if j in used:
                continue
            r = _ratio(xw, tw)
            if r > best:
                best, best_j = r, j
        thr = 0.65 if len(tw) <= 4 else 0.75
        if best < thr:
            return 0.0
        used.add(best_j)
        total += best

    for j, xw in enumerate(x_words):
        if j in used:
            continue
        if sum(1 for c in xw if c.isalpha()) >= 4 or (vocab and xw in vocab):
            return 0.0

    return total / len(t_words)


def match_filled_container(title: str, desc: str, base: str, fill: str,
                           vocab: frozenset | None = None) -> float:
    """Match a filled container: title must name base, desc must name fill."""
    if not desc:
        return 0.0
    base_score = _word_match_score(title, base, vocab)
    if base_score <= 0.0:
        return 0.0
    fill_score = _word_match_score(desc, fill, vocab)
    if fill_score <= 0.0:
        return 0.0
    return (base_score + fill_score) / 2


def match_battery(title: str, prefix: str, full_target: str) -> float:
    """Match a battery: require exact 2-char prefix, then fuzzy on full name."""
    title_prefix_match = re.match(r'^([a-z]{2})\s+', title)
    if title_prefix_match:
        if title_prefix_match.group(1) != prefix:
            return 0.0
    r = _ratio(title, full_target)
    return r if r >= 0.85 else 0.0


def match_graded(title: str, base: str, grade: str) -> float:
    """Match a graded item: require exact bracket content, fuzzy on base name."""
    title_bracket = re.search(r'\[(.+?)\]', title)
    if title_bracket:
        if title_bracket.group(1).strip().lower() != grade.lower():
            return 0.0
        title_base = re.sub(r'\s*\[.+?\]', '', title).strip()
        return _word_match_score(title_base, base)  # vocab not needed: bracket grade already gated
    # No bracket found in OCR - require high overall similarity
    full_target = f"{base} [{grade}]"
    r = _ratio(title, full_target)
    return r if r >= 0.85 else 0.0


def match_simple(title: str, target_name: str, vocab: frozenset | None = None) -> float:
    """Match a simple item via word coverage.

    Requires both title and target to be at least 3 chars to avoid
    garbage matches like 'w' or '—' matching real item names.
    """
    if len(title) < 3 or len(target_name) < 3:
        return 0.0
    return _word_match_score(title, target_name, vocab)


def _is_garbage(text: str) -> bool:
    """Return True if text is likely OCR garbage (symbols, single chars, etc.)."""
    if len(text) < 3:
        return True
    alpha_count = sum(1 for c in text if c.isalpha())
    if alpha_count < 2:
        return True
    return False


def match_score(ocr_text: str, target: str, vocab: frozenset | None = None) -> float:
    """Score how well OCR text identifies a target item name.

    Args:
        ocr_text: Lowercase OCR output, potentially "title | description" format
        target: Lowercase target item name from ITEM_GAME_ENGLISH
        vocab: Optional set of all words appearing in known item names,
               used to tell real leftover words from OCR junk

    Returns 0.0 if the OCR text does not identify the target, else a
    similarity score in (0, 1] - higher is a closer match.
    """
    if not ocr_text or not target:
        return 0.0

    title, desc = parse_ocr_text(ocr_text)

    # Reject garbage OCR output before attempting match
    if _is_garbage(title):
        return 0.0

    item_type, components = classify_target(target)

    if item_type == ItemType.FILLED_CONTAINER:
        return match_filled_container(title, desc, components["base"], components["fill"], vocab)

    # A tooltip with a "Filled with X" description is a filled container -
    # it can never be a simple/battery/graded item ('Cryston Bottle |
    # Filled with Y' must not match the plain 'cryston bottle' item).
    if desc:
        return 0.0

    if item_type == ItemType.BATTERY:
        return match_battery(title, components["prefix"], components["rest"])

    if item_type == ItemType.GRADED:
        return match_graded(title, components["base"], components["grade"])

    return match_simple(title, components["name"], vocab)


def matches(ocr_text: str, target: str, vocab: frozenset | None = None) -> bool:
    """Check if OCR text matches a target item name."""
    return match_score(ocr_text, target, vocab) > 0.0


def build_vocab(item_names) -> frozenset:
    """Build the word vocabulary from an iterable of item names."""
    words = set()
    for name in item_names:
        for w in re.sub(r'[()\[\]]', ' ', name.lower()).split():
            words.add(w)
    return frozenset(words)
