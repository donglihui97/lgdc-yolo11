"""Paper names (U1-U12) of the removable units of YOLO11n (no torch dependency)."""

# Unit names in the paper (U1-U12) and the module indices of YOLO11n they refer to.
UNIT_IDS = {
    'U1': 'c3k2.2', 'U2': 'c3k2.4', 'U3': 'c3k2.6', 'U4': 'c3k2.8', 'U5': 'sppf.9', 'U6': 'c2psa.10',
    'U7': 'c3k2.13', 'U8': 'c3k2.16', 'U9': 'c3k2.19', 'U10': 'c3k2.22', 'U11': 'head.reg', 'U12': 'head.cls',
}
PAPER_NAME = {v: k for k, v in UNIT_IDS.items()}


def parse_units(text):
    """'U4,U7' or 'c3k2.8,c3k2.13' -> ['c3k2.8', 'c3k2.13']; 'none' -> []."""
    if text in (None, '', 'none'):
        return []
    return [UNIT_IDS.get(u.strip(), u.strip()) for u in text.split(',')]
