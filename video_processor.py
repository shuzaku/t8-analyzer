"""
Video processor for analyzing Tekken 8 videos.
Processes video frame by frame and tracks UI elements per player.

Player attribution works by scanning the full center hit-popup zone once per
frame.  The detector (UIDetector) splits OCR tokens at the frame midpoint and
runs pattern matching independently per side, so both players can score the
same element type simultaneously (e.g. both land COUNTER HIT in the same
frame).  Each detected element's absolute x-center determines ownership:
  left of mid_x  → P1
  right of mid_x → P2
Results are deduplicated by (player, element) before the cooldown check.
"""

import cv2
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
from ui_detector import UIDetector


class VideoProcessor:
    """Processes video files and tracks UI elements, assigning each event to P1 or P2."""

    def __init__(
        self,
        detector: UIDetector,
        frame_skip: int = 1,
        cooldown_seconds: float = 2.0,
        full_frame: bool = False,
    ):
        """
        Initialize video processor.

        Args:
            detector: UIDetector (or RoboflowDetector) instance
            frame_skip: Process every Nth frame (1 = all frames)
            cooldown_seconds: Ignore the same (player, element) pair for this many seconds
            full_frame: If True, scan full frame instead of center hit-popup zone
        """
        self.detector = detector
        self.frame_skip = frame_skip
        self.cooldown_seconds = cooldown_seconds
        self.full_frame = full_frame

    def process_video(self, video_path: str, progress_callback: Optional[callable] = None) -> Dict:
        """
        Process a video file and track UI elements per player.

        The full center hit-popup zone is scanned once per frame.  UIDetector
        internally splits OCR tokens by screen side and runs matching per side,
        so the same element can be returned for both players in one call.
        Each detection's abs_x < mid_x → P1, abs_x >= mid_x → P2.
        Results are deduplicated by (player, element) before the cooldown check.

        Args:
            video_path: Path to the video file
            progress_callback: Optional callback(frame_number, total_frames, video_time, total_events)

        Returns:
            Dictionary with per-player stats, flat event list, and video metadata
        """
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Processing video: {video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps:.2f}, Resolution: {width}x{height}")

        # When a compound event fires, also cool down its sub-events for the same player.
        # This prevents partial OCR reads (e.g. "HEAT" when "HEAT BURST" was the real event)
        # from producing spurious standalone sub-event detections.
        SUPERSEDES: Dict[str, List[str]] = {
            "HEAT BURST":    ["HEAT"],
            "HEAT SMASH":    ["HEAT"],
            "HEAT ENGAGER":  ["HEAT"],
            "RAGE ART":      ["RAGE DRIVE"],
            "WALL BREAK":    ["WALL SPLAT"],
        }

        stats: Dict[str, Dict[str, int]] = {"P1": defaultdict(int), "P2": defaultdict(int)}
        frame_detections: List[Dict] = []
        last_seen_time: Dict[tuple, float] = {}

        # Scan zone: full width, skip the top HUD bar (~15%) and bottom health/timer UI (~10%).
        # Tekken 8 hit popups appear near each character (left/right), so no x-margin is used.
        # UIDetector splits OCR tokens at the frame midpoint internally.
        top_margin    = int(height * 0.15)
        bottom_margin = int(height * 0.10)
        hit_zone      = (0, top_margin, width, height - top_margin - bottom_margin)

        # Vertical center line dividing P1 (left) from P2 (right)
        mid_x = width / 2.0

        frame_number = 0
        processed_frames = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_number % self.frame_skip != 0:
                frame_number += 1
                continue

            current_time = frame_number / fps

            if self.full_frame:
                detections = self.detector.detect_in_region_positioned(
                    frame, (0, 0, width, height)
                )
            else:
                detections = self.detector.detect_in_region_positioned(frame, hit_zone)

            # Deduplicate by (player, element) — keeps first occurrence per pair.
            seen_player_elements: dict = {}
            for element, abs_x in detections:
                player = "P1" if abs_x < mid_x else "P2"
                key = (player, element)
                if key not in seen_player_elements:
                    seen_player_elements[key] = abs_x

            for (player, element), abs_x in seen_player_elements.items():
                key = (player, element)
                last = last_seen_time.get(key, -999.0)
                if current_time - last >= self.cooldown_seconds:
                    stats[player][element] += 1
                    last_seen_time[key] = current_time
                    frame_detections.append({
                        "frame": frame_number,
                        "time": current_time,
                        "player": player,
                        "element": element,
                    })
                    for sub in SUPERSEDES.get(element, []):
                        last_seen_time[(player, sub)] = current_time

            processed_frames += 1
            frame_number += 1

            if progress_callback and processed_frames % 5 == 0:
                total_events = sum(sum(s.values()) for s in stats.values())
                progress_callback(frame_number, total_frames, current_time, total_events)

        cap.release()

        duration = total_frames / fps if fps > 0 else 0

        return {
            "stats": {p: dict(s) for p, s in stats.items()},
            "frame_detections": frame_detections,
            "video_info": {
                "total_frames": total_frames,
                "processed_frames": processed_frames,
                "frame_skip": self.frame_skip,
                "fps": fps,
                "duration": duration,
                "resolution": (width, height),
            },
        }
