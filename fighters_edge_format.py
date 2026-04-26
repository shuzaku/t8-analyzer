"""
Fighters-Edge analyses shape: MatchType, MatchId, Detections[{label, player, timestamp}],
plus wide CSV export (no bboxes).
"""

import csv
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def element_to_label(element: str) -> str:
    """PUNISH COUNTER -> punish_counter"""
    return (
        element.strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def player_to_edge(player: str) -> str:
    """P1 -> p1"""
    return player.strip().lower()


def build_detections_list(frame_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Build Detections array from analyzer frame_detections (or to_dict frame_detections).

    Each item: label, player, timestamp (seconds, int). No bboxes.
    """
    out: List[Dict[str, Any]] = []
    for e in frame_events:
        t = e.get("time_seconds")
        if t is None:
            t = e.get("time")
        ts = int(round(float(t))) if t is not None else 0
        out.append(
            {
                "label": element_to_label(str(e["element"])),
                "player": player_to_edge(str(e["player"])),
                "timestamp": ts,
            }
        )
    return out


def export_fighters_edge_csv(
    output_path: Path,
    *,
    row_id: str,
    match_type: str,
    match_id: Optional[str],
    detections: List[Dict[str, Any]],
    min_slots: int = 69,
) -> None:
    """Write one row: _id, MatchType, MatchId, then per-slot label / player / timestamp (no bboxes)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n = max(len(detections), min_slots)

    header: List[str] = ["_id", "MatchType", "MatchId"]
    for i in range(n):
        header.append(f"Detections[{i}].label")
    for i in range(n):
        header.append(f"Detections[{i}].player")
    for i in range(n):
        header.append(f"Detections[{i}].timestamp")

    row: List[Any] = [row_id, match_type, match_id or ""]

    for i in range(n):
        row.append(detections[i]["label"] if i < len(detections) else "")

    for i in range(n):
        row.append(detections[i]["player"] if i < len(detections) else "")

    for i in range(n):
        row.append(detections[i]["timestamp"] if i < len(detections) else "")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


_YT_ID_RE = re.compile(
    r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/|youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
    re.IGNORECASE,
)


def extract_youtube_video_id(url: str) -> Optional[str]:
    m = _YT_ID_RE.search(url)
    return m.group(1) if m else None


def video_url_match_variants(url: str) -> List[str]:
    """URLs to try for exact match against stored videoUrl.

    Tries the bare video ID first (e.g. "0UcW1VD37RI") since that is the
    format used in the Fighters-Edge matches collection, then falls back to
    full URL forms.
    """
    u = url.strip()
    vid = extract_youtube_video_id(u)
    variants = []
    # Bare video ID first — matches documents that store just the ID
    if vid:
        variants.append(vid)
    variants.append(u)
    if u.endswith("/"):
        variants.append(u.rstrip("/"))
    if vid:
        variants.append(f"https://www.youtube.com/watch?v={vid}")
        variants.append(f"https://youtu.be/{vid}")
    # de-dupe preserving order
    seen = set()
    out = []
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out
