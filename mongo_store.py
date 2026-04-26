"""
Save analysis results to MongoDB (Fighters-Edge shape + match lookup).
Configure with MONGO_URI, MONGO_DATABASE, MONGO_COLLECTION, MONGO_MATCHES_COLLECTION, MONGO_VIDEO_URL_FIELD.
"""

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fighters_edge_format import (
    build_detections_list,
    extract_youtube_video_id,
    video_url_match_variants,
)


def find_match_by_video_url(
    client,
    db_name: str,
    video_url: str,
) -> Optional[Dict[str, Any]]:
    """
    Find a document in the matches collection whose video URL field matches the given URL.
    Tries exact variants (youtu.be vs watch) then regex on YouTube video id.
    """
    if not video_url or not video_url.strip():
        return None

    field = os.environ.get("MONGO_VIDEO_URL_FIELD", "videoUrl").strip()
    coll_name = os.environ.get("MONGO_MATCHES_COLLECTION", "matches").strip()
    coll = client[db_name][coll_name]

    for v in video_url_match_variants(video_url):
        doc = coll.find_one({field: v})
        if doc:
            return doc

    vid = extract_youtube_video_id(video_url)
    if vid:
        doc = coll.find_one({field: {"$regex": re.escape(vid)}})
        if doc:
            return doc

    return None


def _resolve_match_fields(match_doc: Optional[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    """Return (match_id_str, match_type) from a matches collection document."""
    if not match_doc:
        return None, "Unknown"

    mid = match_doc.get("matchId") or match_doc.get("MatchId")
    if mid is not None:
        return str(mid), str(
            match_doc.get("MatchType")
            or match_doc.get("matchType")
            or "Tournament"
        )

    oid = match_doc.get("_id")
    if oid is not None:
        return str(oid), str(
            match_doc.get("MatchType")
            or match_doc.get("matchType")
            or "Tournament"
        )

    return None, "Unknown"


def _match_id_for_bson(match_id_str: Optional[str]):
    """Use BSON ObjectId when match_id is 24 hex chars."""
    if not match_id_str:
        return None
    try:
        from bson import ObjectId
        if len(match_id_str) == 24 and re.match(r"^[a-fA-F0-9]{24}$", match_id_str):
            return ObjectId(match_id_str)
    except Exception:
        pass
    return match_id_str


def save_analysis(
    payload: Dict[str, Any],
    *,
    video_path: str,
    detector: str,
    detector_detail: Optional[str] = None,
    video_url: Optional[str] = None,
    require_match: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Look up MatchId in `matches` by videoUrl, then insert an analysis document.

    Document includes Fighters-Edge fields: MatchType, MatchId, videoUrl, Detections[]
    plus analyzer payload (summary, video_info, player_stats, frame_detections).

    Args:
        video_url: YouTube (or other) URL used to find the match; local-only runs can omit.
        require_match: If True and video_url is set, raises ValueError when no match is found.

    Returns:
        Dict with inserted_id, match_id, match_type, match_found, detections — or None if Mongo disabled.
    """
    uri = os.environ.get("MONGO_URI", "").strip()
    if not uri:
        return None

    try:
        from pymongo import MongoClient
    except ImportError:
        raise ImportError("MongoDB support requires pymongo. Install with: pip install pymongo")

    database_name = os.environ.get("MONGO_DATABASE", "tekken8_analyzer").strip()
    collection_name = os.environ.get("MONGO_COLLECTION", "analyses").strip()

    video_path_resolved = str(Path(video_path).resolve())
    frame_events = payload.get("frame_detections") or []
    detections_list = build_detections_list(frame_events)

    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except Exception as e:
        client.close()
        raise ConnectionError(f"MongoDB connection failed: {e}") from e

    match_doc = None
    if video_url and video_url.strip():
        match_doc = find_match_by_video_url(client, database_name, video_url.strip())

    match_id_str, match_type = _resolve_match_fields(match_doc)

    if require_match and video_url and video_url.strip() and not match_id_str:
        client.close()
        raise ValueError(
            f"No match found in collection '{os.environ.get('MONGO_MATCHES_COLLECTION', 'matches')}' "
            f"for videoUrl matching: {video_url!r}"
        )

    match_id_bson = _match_id_for_bson(match_id_str)

    doc: Dict[str, Any] = {
        **payload,
        "analyzed_at": datetime.now(timezone.utc),
        "MatchType": match_type,
        "MatchId": match_id_bson,
        "match_lookup_found": bool(match_doc),
        "videoUrl": (video_url.strip() if video_url else None),
        "Detections": detections_list,
        "video_path": video_path_resolved,
        "detector": detector,
        "detector_detail": detector_detail,
    }

    coll = client[database_name][collection_name]
    result = coll.insert_one(doc)
    inserted_id = str(result.inserted_id)
    client.close()

    return {
        "inserted_id": inserted_id,
        "match_id": match_id_str,
        "match_type": match_type,
        "match_found": bool(match_doc),
        "detections": detections_list,
    }
